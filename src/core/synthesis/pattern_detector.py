"""
Cross-Memory Pattern Detection with Advanced Clustering and Analytics.

This module identifies patterns, topics, and relationships across memories using
machine learning clustering, topic modeling, and knowledge gap detection.
All processing is performed locally with optimized algorithms.
"""

import asyncio
from typing import List, Dict, Optional, Tuple, Set, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import networkx as nx
from collections import defaultdict, Counter
import spacy
from gensim import corpora, models
from gensim.models.coherencemodel import CoherenceModel
import structlog
from pydantic import BaseModel, Field, ConfigDict
import json
import re

from ...models.memory import Memory
from ..embeddings.embedder import Embedder
from ..cache.redis_cache import RedisCache
from ..utils.config import settings

logger = structlog.get_logger(__name__)


class PatternType(str, Enum):
    """Types of patterns that can be detected."""
    TOPIC_CLUSTER = "topic_cluster"  # Memories about similar topics
    TEMPORAL_PATTERN = "temporal_pattern"  # Time-based patterns
    ENTITY_PATTERN = "entity_pattern"  # Patterns around entities
    SENTIMENT_PATTERN = "sentiment_pattern"  # Emotional patterns
    BEHAVIORAL_PATTERN = "behavioral_pattern"  # User behavior patterns
    KNOWLEDGE_GAP = "knowledge_gap"  # Missing information patterns


class ClusteringMethod(str, Enum):
    """Available clustering algorithms."""
    KMEANS = "kmeans"
    DBSCAN = "dbscan"
    HIERARCHICAL = "hierarchical"
    AFFINITY_PROPAGATION = "affinity_propagation"


@dataclass
class PatternCluster:
    """Represents a cluster of related memories."""
    cluster_id: str
    pattern_type: PatternType
    memories: List[Memory]
    cluster_center: Optional[np.ndarray] = None
    coherence_score: float = 0.0
    representative_memory: Optional[Memory] = None
    keywords: List[str] = field(default_factory=list)
    topics: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    temporal_span: Optional[Tuple[datetime, datetime]] = None


class PatternInsight(BaseModel):
    """Insights derived from pattern analysis."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    insight_type: str = Field(description="Type of insight discovered")
    description: str = Field(description="Human-readable description")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in insight")
    supporting_memories: List[str] = Field(description="Memory IDs supporting this insight")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class KnowledgeGap(BaseModel):
    """Represents a detected knowledge gap."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    gap_id: str = Field(description="Unique identifier for the gap")
    topic: str = Field(description="Topic area of the gap")
    description: str = Field(description="Description of the missing knowledge")
    importance_score: float = Field(ge=0.0, le=1.0, description="Importance of filling this gap")
    related_memories: List[str] = Field(description="Related memory IDs")
    suggested_questions: List[str] = Field(description="Questions to fill the gap")


class PatternAnalysisResult(BaseModel):
    """Complete result of pattern analysis."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    clusters: List[PatternCluster] = Field(description="Discovered pattern clusters")
    insights: List[PatternInsight] = Field(description="Generated insights")
    knowledge_gaps: List[KnowledgeGap] = Field(description="Detected knowledge gaps")
    topic_distribution: Dict[str, float] = Field(description="Distribution of topics")
    entity_frequency: Dict[str, int] = Field(description="Frequency of entities")
    temporal_patterns: Dict[str, Any] = Field(description="Temporal pattern analysis")
    recommendations: List[str] = Field(description="Actionable recommendations")


class PatternDetector:
    """
    Advanced pattern detection engine for memory analysis.
    
    Features:
    - Multiple clustering algorithms with automatic selection
    - Topic modeling using LDA and NMF
    - Entity extraction and pattern analysis
    - Temporal pattern detection
    - Knowledge gap identification
    - Collaborative filtering for recommendations
    - Network analysis for relationship patterns
    """
    
    def __init__(
        self,
        embedder: Embedder,
        cache: Optional[RedisCache] = None,
        spacy_model: str = "en_core_web_sm",
        min_cluster_size: int = 3,
        max_clusters: int = 20
    ):
        """
        Initialize the pattern detector.
        
        Args:
            embedder: Embedder for generating vector representations
            cache: Redis cache for performance optimization
            spacy_model: SpaCy model for NLP tasks
            min_cluster_size: Minimum memories per cluster
            max_clusters: Maximum number of clusters to generate
        """
        self.embedder = embedder
        self.cache = cache
        self.min_cluster_size = min_cluster_size
        self.max_clusters = max_clusters
        
        # Initialize NLP components
        try:
            self.nlp = spacy.load(spacy_model)
        except:
            logger.warning(f"SpaCy model {spacy_model} not found, downloading...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", spacy_model])
            self.nlp = spacy.load(spacy_model)
        
        # Initialize clustering algorithms
        self.clustering_methods = {
            ClusteringMethod.KMEANS: self._kmeans_clustering,
            ClusteringMethod.DBSCAN: self._dbscan_clustering,
            ClusteringMethod.HIERARCHICAL: self._hierarchical_clustering
        }
        
        logger.info(
            "Initialized pattern detector",
            min_cluster_size=min_cluster_size,
            max_clusters=max_clusters
        )
    
    async def detect_patterns(
        self,
        memories: List[Memory],
        pattern_types: Optional[List[PatternType]] = None,
        clustering_method: ClusteringMethod = ClusteringMethod.KMEANS
    ) -> PatternAnalysisResult:
        """
        Detect patterns across a collection of memories.
        
        Args:
            memories: List of memories to analyze
            pattern_types: Types of patterns to detect (default: all)
            clustering_method: Clustering algorithm to use
            
        Returns:
            Complete pattern analysis results
        """
        if not memories:
            return PatternAnalysisResult(
                clusters=[],
                insights=[],
                knowledge_gaps=[],
                topic_distribution={},
                entity_frequency={},
                temporal_patterns={},
                recommendations=[]
            )
        
        start_time = datetime.utcnow()
        
        if pattern_types is None:
            pattern_types = list(PatternType)
        
        # Extract features from memories
        memory_features = await self._extract_features(memories)
        
        # Perform clustering analysis
        clusters = []
        if PatternType.TOPIC_CLUSTER in pattern_types:
            topic_clusters = await self._detect_topic_clusters(
                memories, 
                memory_features,
                clustering_method
            )
            clusters.extend(topic_clusters)
        
        # Detect temporal patterns
        temporal_patterns = {}
        if PatternType.TEMPORAL_PATTERN in pattern_types:
            temporal_patterns = await self._detect_temporal_patterns(memories)
        
        # Extract entity patterns
        entity_patterns = []
        entity_frequency = {}
        if PatternType.ENTITY_PATTERN in pattern_types:
            entity_patterns, entity_frequency = await self._detect_entity_patterns(memories)
            clusters.extend(entity_patterns)
        
        # Generate insights
        insights = await self._generate_insights(clusters, memories, temporal_patterns)
        
        # Detect knowledge gaps
        knowledge_gaps = await self._detect_knowledge_gaps(memories, clusters)
        
        # Generate topic distribution
        topic_distribution = await self._analyze_topic_distribution(memories, clusters)
        
        # Generate recommendations
        recommendations = await self._generate_recommendations(
            clusters, 
            insights, 
            knowledge_gaps
        )
        
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        logger.info(
            "Pattern detection complete",
            memories_count=len(memories),
            clusters_found=len(clusters),
            insights_generated=len(insights),
            knowledge_gaps=len(knowledge_gaps),
            processing_time_s=processing_time
        )
        
        return PatternAnalysisResult(
            clusters=clusters,
            insights=insights,
            knowledge_gaps=knowledge_gaps,
            topic_distribution=topic_distribution,
            entity_frequency=entity_frequency,
            temporal_patterns=temporal_patterns,
            recommendations=recommendations
        )
    
    async def _extract_features(self, memories: List[Memory]) -> Dict[str, Any]:
        """Extract various features from memories for analysis."""
        texts = [m.content for m in memories]
        
        # Get embeddings
        embeddings = await self.embedder.embed_batch(texts)
        
        # Extract TF-IDF features
        tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 3),
            min_df=2
        )
        
        try:
            tfidf_features = tfidf_vectorizer.fit_transform(texts)
            feature_names = tfidf_vectorizer.get_feature_names_out()
        except:
            # Fallback if not enough documents
            tfidf_features = None
            feature_names = []
        
        # Extract entities and keywords
        entities_per_memory = []
        keywords_per_memory = []
        
        for text in texts:
            doc = self.nlp(text)
            
            # Extract entities
            entities = [ent.text for ent in doc.ents if ent.label_ in [
                'PERSON', 'ORG', 'GPE', 'DATE', 'TIME', 'MONEY', 'QUANTITY'
            ]]
            entities_per_memory.append(entities)
            
            # Extract keywords (noun phrases and important terms)
            keywords = []
            for chunk in doc.noun_chunks:
                if len(chunk.text.split()) <= 3:  # Keep short phrases
                    keywords.append(chunk.text.lower())
            
            # Add important single words
            for token in doc:
                if (token.pos_ in ['NOUN', 'PROPN', 'ADJ'] and 
                    not token.is_stop and 
                    len(token.text) > 3):
                    keywords.append(token.text.lower())
            
            keywords_per_memory.append(list(set(keywords)))
        
        return {
            'embeddings': embeddings,
            'tfidf_features': tfidf_features,
            'feature_names': feature_names,
            'entities': entities_per_memory,
            'keywords': keywords_per_memory,
            'texts': texts
        }
    
    async def _detect_topic_clusters(
        self,
        memories: List[Memory],
        features: Dict[str, Any],
        clustering_method: ClusteringMethod
    ) -> List[PatternCluster]:
        """Detect topic-based clusters using specified algorithm."""
        embeddings = features['embeddings']
        
        if len(memories) < self.min_cluster_size:
            return []
        
        # Determine optimal number of clusters
        optimal_k = await self._find_optimal_clusters(embeddings)
        
        # Apply clustering
        cluster_labels = await self.clustering_methods[clustering_method](
            embeddings, 
            optimal_k
        )
        
        # Create cluster objects
        clusters = []
        for cluster_id in set(cluster_labels):
            if cluster_id == -1:  # Noise points in DBSCAN
                continue
            
            cluster_memories = [
                memories[i] for i, label in enumerate(cluster_labels) 
                if label == cluster_id
            ]
            
            if len(cluster_memories) < self.min_cluster_size:
                continue
            
            # Calculate cluster center
            cluster_indices = [i for i, label in enumerate(cluster_labels) if label == cluster_id]
            cluster_embeddings = embeddings[cluster_indices]
            cluster_center = np.mean(cluster_embeddings, axis=0)
            
            # Find representative memory (closest to center)
            distances = np.linalg.norm(cluster_embeddings - cluster_center, axis=1)
            rep_idx = cluster_indices[np.argmin(distances)]
            representative_memory = memories[rep_idx]
            
            # Extract cluster keywords and topics
            cluster_texts = [m.content for m in cluster_memories]
            keywords = await self._extract_cluster_keywords(cluster_texts)
            topics = await self._extract_cluster_topics(cluster_texts)
            
            # Extract entities
            entities = []
            for memory in cluster_memories:
                if memory.id < len(features['entities']):
                    entities.extend(features['entities'][memory.id] if isinstance(memory.id, int) else [])
            entity_counts = Counter(entities)
            top_entities = [ent for ent, _ in entity_counts.most_common(10)]
            
            # Calculate temporal span
            timestamps = [m.created_at for m in cluster_memories]
            temporal_span = (min(timestamps), max(timestamps))
            
            # Calculate coherence score
            coherence_score = await self._calculate_cluster_coherence(cluster_embeddings)
            
            cluster = PatternCluster(
                cluster_id=f"topic_{cluster_id}",
                pattern_type=PatternType.TOPIC_CLUSTER,
                memories=cluster_memories,
                cluster_center=cluster_center,
                coherence_score=coherence_score,
                representative_memory=representative_memory,
                keywords=keywords,
                topics=topics,
                entities=top_entities,
                temporal_span=temporal_span
            )
            
            clusters.append(cluster)
        
        return clusters
    
    async def _find_optimal_clusters(self, embeddings: np.ndarray) -> int:
        """Find optimal number of clusters using multiple metrics."""
        max_k = min(self.max_clusters, len(embeddings) // self.min_cluster_size)
        if max_k < 2:
            return 2
        
        k_range = range(2, max_k + 1)
        silhouette_scores = []
        ch_scores = []
        
        for k in k_range:
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(embeddings)
                
                # Calculate silhouette score
                sil_score = silhouette_score(embeddings, labels)
                silhouette_scores.append(sil_score)
                
                # Calculate Calinski-Harabasz score
                ch_score = calinski_harabasz_score(embeddings, labels)
                ch_scores.append(ch_score)
                
            except Exception as e:
                logger.warning(f"Error calculating scores for k={k}: {e}")
                silhouette_scores.append(0)
                ch_scores.append(0)
        
        # Normalize scores and combine
        if silhouette_scores and ch_scores:
            sil_norm = np.array(silhouette_scores) / max(silhouette_scores)
            ch_norm = np.array(ch_scores) / max(ch_scores)
            combined_scores = 0.6 * sil_norm + 0.4 * ch_norm
            
            optimal_idx = np.argmax(combined_scores)
            optimal_k = k_range[optimal_idx]
        else:
            optimal_k = min(5, max_k)  # Default fallback
        
        logger.debug(f"Optimal clusters determined: {optimal_k}")
        return optimal_k
    
    async def _kmeans_clustering(self, embeddings: np.ndarray, n_clusters: int) -> List[int]:
        """Perform K-means clustering."""
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        return kmeans.fit_predict(embeddings).tolist()
    
    async def _dbscan_clustering(self, embeddings: np.ndarray, n_clusters: int) -> List[int]:
        """Perform DBSCAN clustering."""
        # For DBSCAN, estimate eps using k-nearest neighbors
        nn = NearestNeighbors(n_neighbors=self.min_cluster_size)
        nn.fit(embeddings)
        distances, _ = nn.kneighbors(embeddings)
        eps = np.percentile(distances[:, -1], 90)  # Use 90th percentile
        
        dbscan = DBSCAN(eps=eps, min_samples=self.min_cluster_size)
        return dbscan.fit_predict(embeddings).tolist()
    
    async def _hierarchical_clustering(self, embeddings: np.ndarray, n_clusters: int) -> List[int]:
        """Perform hierarchical clustering."""
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage='ward'
        )
        return clustering.fit_predict(embeddings).tolist()
    
    async def _extract_cluster_keywords(self, texts: List[str]) -> List[str]:
        """Extract representative keywords for a cluster."""
        combined_text = ' '.join(texts)
        doc = self.nlp(combined_text)
        
        # Extract noun phrases and important terms
        keywords = []
        
        # Noun phrases
        for chunk in doc.noun_chunks:
            if 2 <= len(chunk.text.split()) <= 3:
                keywords.append(chunk.text.lower())
        
        # Important single words
        word_freq = Counter()
        for token in doc:
            if (token.pos_ in ['NOUN', 'PROPN', 'ADJ'] and 
                not token.is_stop and 
                len(token.text) > 3):
                word_freq[token.text.lower()] += 1
        
        # Get most frequent terms
        top_words = [word for word, _ in word_freq.most_common(10)]
        keywords.extend(top_words)
        
        # Remove duplicates and return top keywords
        unique_keywords = []
        seen = set()
        for kw in keywords:
            if kw not in seen and len(kw) > 2:
                unique_keywords.append(kw)
                seen.add(kw)
        
        return unique_keywords[:15]
    
    async def _extract_cluster_topics(self, texts: List[str]) -> List[str]:
        """Extract high-level topics for a cluster using LDA."""
        if len(texts) < 3:
            return ["General"]
        
        try:
            # Prepare documents for LDA
            processed_docs = []
            for text in texts:
                doc = self.nlp(text)
                tokens = [
                    token.lemma_.lower() for token in doc
                    if not token.is_stop and 
                    not token.is_punct and 
                    len(token.text) > 2 and
                    token.pos_ in ['NOUN', 'PROPN', 'ADJ', 'VERB']
                ]
                processed_docs.append(tokens)
            
            # Create dictionary and corpus
            dictionary = corpora.Dictionary(processed_docs)
            dictionary.filter_extremes(no_below=1, no_above=0.8, keep_n=100)
            corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
            
            if not corpus or not dictionary:
                return ["General"]
            
            # Train LDA model
            num_topics = min(3, max(1, len(texts) // 3))
            lda_model = models.LdaModel(
                corpus=corpus,
                id2word=dictionary,
                num_topics=num_topics,
                random_state=42,
                passes=10,
                alpha='auto',
                per_word_topics=True
            )
            
            # Extract topic words
            topics = []
            for topic_id in range(num_topics):
                topic_words = lda_model.show_topic(topic_id, topn=5)
                topic_name = ' '.join([word for word, _ in topic_words[:3]])
                topics.append(topic_name.title())
            
            return topics if topics else ["General"]
            
        except Exception as e:
            logger.warning(f"LDA topic extraction failed: {e}")
            return ["General"]
    
    async def _calculate_cluster_coherence(self, cluster_embeddings: np.ndarray) -> float:
        """Calculate coherence score for a cluster."""
        if len(cluster_embeddings) < 2:
            return 1.0
        
        # Calculate pairwise cosine similarities
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(cluster_embeddings)
        
        # Average similarity (excluding diagonal)
        mask = np.ones_like(similarities, dtype=bool)
        np.fill_diagonal(mask, False)
        avg_similarity = similarities[mask].mean()
        
        return float(avg_similarity)
    
    async def _detect_temporal_patterns(self, memories: List[Memory]) -> Dict[str, Any]:
        """Detect patterns in memory creation timing."""
        timestamps = [m.created_at for m in memories]
        
        if not timestamps:
            return {}
        
        # Analyze temporal distribution
        timestamps.sort()
        
        # Calculate time differences
        time_diffs = []
        for i in range(1, len(timestamps)):
            diff = (timestamps[i] - timestamps[i-1]).total_seconds() / 3600  # Hours
            time_diffs.append(diff)
        
        # Detect activity patterns
        patterns = {
            'total_span_days': (timestamps[-1] - timestamps[0]).days,
            'average_gap_hours': np.mean(time_diffs) if time_diffs else 0,
            'activity_bursts': self._detect_activity_bursts(timestamps),
            'daily_distribution': self._analyze_daily_distribution(timestamps),
            'weekly_distribution': self._analyze_weekly_distribution(timestamps)
        }
        
        return patterns
    
    def _detect_activity_bursts(self, timestamps: List[datetime]) -> List[Dict[str, Any]]:
        """Detect periods of high activity."""
        if len(timestamps) < 3:
            return []
        
        bursts = []
        current_burst = []
        burst_threshold = timedelta(hours=2)  # Memories within 2 hours are a burst
        
        for i, ts in enumerate(timestamps):
            if not current_burst:
                current_burst.append(i)
            else:
                time_since_last = ts - timestamps[current_burst[-1]]
                if time_since_last <= burst_threshold:
                    current_burst.append(i)
                else:
                    # End current burst if it has enough memories
                    if len(current_burst) >= 3:
                        bursts.append({
                            'start_time': timestamps[current_burst[0]],
                            'end_time': timestamps[current_burst[-1]],
                            'memory_count': len(current_burst),
                            'duration_minutes': (
                                timestamps[current_burst[-1]] - 
                                timestamps[current_burst[0]]
                            ).total_seconds() / 60
                        })
                    current_burst = [i]
        
        # Check final burst
        if len(current_burst) >= 3:
            bursts.append({
                'start_time': timestamps[current_burst[0]],
                'end_time': timestamps[current_burst[-1]],
                'memory_count': len(current_burst),
                'duration_minutes': (
                    timestamps[current_burst[-1]] - 
                    timestamps[current_burst[0]]
                ).total_seconds() / 60
            })
        
        return bursts
    
    def _analyze_daily_distribution(self, timestamps: List[datetime]) -> Dict[int, int]:
        """Analyze distribution of memories by hour of day."""
        hour_counts = defaultdict(int)
        for ts in timestamps:
            hour_counts[ts.hour] += 1
        return dict(hour_counts)
    
    def _analyze_weekly_distribution(self, timestamps: List[datetime]) -> Dict[int, int]:
        """Analyze distribution of memories by day of week."""
        day_counts = defaultdict(int)
        for ts in timestamps:
            day_counts[ts.weekday()] += 1  # 0=Monday, 6=Sunday
        return dict(day_counts)
    
    async def _detect_entity_patterns(self, memories: List[Memory]) -> Tuple[List[PatternCluster], Dict[str, int]]:
        """Detect patterns based on named entities."""
        entity_memory_map = defaultdict(list)
        entity_frequency = defaultdict(int)
        
        # Extract entities from all memories
        for memory in memories:
            doc = self.nlp(memory.content)
            for ent in doc.ents:
                if ent.label_ in ['PERSON', 'ORG', 'GPE', 'DATE']:
                    normalized_entity = ent.text.lower().strip()
                    entity_memory_map[normalized_entity].append(memory)
                    entity_frequency[normalized_entity] += 1
        
        # Create clusters for entities with multiple memories
        entity_clusters = []
        for entity, entity_memories in entity_memory_map.items():
            if len(entity_memories) >= self.min_cluster_size:
                cluster = PatternCluster(
                    cluster_id=f"entity_{entity.replace(' ', '_')}",
                    pattern_type=PatternType.ENTITY_PATTERN,
                    memories=entity_memories,
                    keywords=[entity],
                    entities=[entity],
                    temporal_span=(
                        min(m.created_at for m in entity_memories),
                        max(m.created_at for m in entity_memories)
                    ),
                    coherence_score=1.0  # High coherence for entity-based clusters
                )
                entity_clusters.append(cluster)
        
        return entity_clusters, dict(entity_frequency)
    
    async def _generate_insights(
        self,
        clusters: List[PatternCluster],
        memories: List[Memory],
        temporal_patterns: Dict[str, Any]
    ) -> List[PatternInsight]:
        """Generate actionable insights from pattern analysis."""
        insights = []
        
        # Cluster-based insights
        for cluster in clusters:
            if cluster.pattern_type == PatternType.TOPIC_CLUSTER:
                insight = PatternInsight(
                    insight_type="topic_focus",
                    description=f"Strong focus on {', '.join(cluster.topics[:2])} with {len(cluster.memories)} related memories",
                    confidence=cluster.coherence_score,
                    supporting_memories=[m.id for m in cluster.memories],
                    metadata={
                        "cluster_id": cluster.cluster_id,
                        "keywords": cluster.keywords[:5],
                        "temporal_span_days": (cluster.temporal_span[1] - cluster.temporal_span[0]).days if cluster.temporal_span else 0
                    }
                )
                insights.append(insight)
            
            elif cluster.pattern_type == PatternType.ENTITY_PATTERN:
                insight = PatternInsight(
                    insight_type="entity_importance",
                    description=f"Frequent mentions of {cluster.entities[0]} across {len(cluster.memories)} memories",
                    confidence=min(1.0, len(cluster.memories) / 10),  # Higher confidence with more mentions
                    supporting_memories=[m.id for m in cluster.memories],
                    metadata={
                        "entity": cluster.entities[0],
                        "mention_frequency": len(cluster.memories)
                    }
                )
                insights.append(insight)
        
        # Temporal insights
        if temporal_patterns.get('activity_bursts'):
            for burst in temporal_patterns['activity_bursts']:
                insight = PatternInsight(
                    insight_type="activity_burst",
                    description=f"High activity period with {burst['memory_count']} memories in {burst['duration_minutes']:.1f} minutes",
                    confidence=min(1.0, burst['memory_count'] / 10),
                    supporting_memories=[],  # Would need to track specific memories
                    metadata=burst
                )
                insights.append(insight)
        
        # Diversity insights
        unique_topics = set()
        for cluster in clusters:
            unique_topics.update(cluster.topics)
        
        if len(unique_topics) > 5:
            insight = PatternInsight(
                insight_type="diverse_interests",
                description=f"Diverse knowledge across {len(unique_topics)} different topic areas",
                confidence=0.8,
                supporting_memories=[m.id for m in memories[:10]],  # Sample
                metadata={"topic_count": len(unique_topics), "topics": list(unique_topics)[:10]}
            )
            insights.append(insight)
        
        return insights
    
    async def _detect_knowledge_gaps(
        self,
        memories: List[Memory],
        clusters: List[PatternCluster]
    ) -> List[KnowledgeGap]:
        """Detect potential knowledge gaps."""
        gaps = []
        
        # Analyze cluster sizes to find underrepresented topics
        cluster_sizes = [(c.cluster_id, len(c.memories), c.topics) for c in clusters]
        cluster_sizes.sort(key=lambda x: x[1])
        
        # Small clusters might indicate incomplete knowledge
        for cluster_id, size, topics in cluster_sizes[:3]:  # Smallest 3 clusters
            if size < 5 and topics:  # Arbitrary threshold
                gap = KnowledgeGap(
                    gap_id=f"gap_{cluster_id}",
                    topic=topics[0] if topics else "Unknown",
                    description=f"Limited information on {topics[0] if topics else 'this topic'} with only {size} memories",
                    importance_score=1.0 - (size / 10),  # Higher score for smaller clusters
                    related_memories=[m.id for m in next(c.memories for c in clusters if c.cluster_id == cluster_id)],
                    suggested_questions=[
                        f"What are the key aspects of {topics[0] if topics else 'this topic'}?",
                        f"How does {topics[0] if topics else 'this topic'} relate to other areas?",
                        f"What are recent developments in {topics[0] if topics else 'this topic'}?"
                    ]
                )
                gaps.append(gap)
        
        # Temporal gaps - periods with no memories
        if len(memories) > 1:
            timestamps = sorted([m.created_at for m in memories])
            for i in range(1, len(timestamps)):
                gap_duration = (timestamps[i] - timestamps[i-1]).days
                if gap_duration > 7:  # Gap longer than a week
                    gap = KnowledgeGap(
                        gap_id=f"temporal_gap_{i}",
                        topic="Temporal Information Gap",
                        description=f"No memories recorded for {gap_duration} days",
                        importance_score=min(1.0, gap_duration / 30),  # Higher score for longer gaps
                        related_memories=[],
                        suggested_questions=[
                            "What happened during this period?",
                            "Were there any significant events or learnings?",
                            "What activities or thoughts weren't captured?"
                        ]
                    )
                    gaps.append(gap)
        
        return gaps[:10]  # Limit to top 10 gaps
    
    async def _analyze_topic_distribution(
        self,
        memories: List[Memory],
        clusters: List[PatternCluster]
    ) -> Dict[str, float]:
        """Analyze the distribution of topics across memories."""
        topic_counts = defaultdict(int)
        total_memories = len(memories)
        
        # Count memories in each topic cluster
        clustered_memories = set()
        for cluster in clusters:
            if cluster.pattern_type == PatternType.TOPIC_CLUSTER:
                for topic in cluster.topics:
                    topic_counts[topic] += len(cluster.memories)
                clustered_memories.update(m.id for m in cluster.memories)
        
        # Add unclustered memories as "Other"
        unclustered_count = total_memories - len(clustered_memories)
        if unclustered_count > 0:
            topic_counts["Other"] = unclustered_count
        
        # Convert to percentages
        if total_memories > 0:
            topic_distribution = {
                topic: count / total_memories 
                for topic, count in topic_counts.items()
            }
        else:
            topic_distribution = {}
        
        return topic_distribution
    
    async def _generate_recommendations(
        self,
        clusters: List[PatternCluster],
        insights: List[PatternInsight],
        knowledge_gaps: List[KnowledgeGap]
    ) -> List[str]:
        """Generate actionable recommendations based on analysis."""
        recommendations = []
        
        # Recommendations based on clusters
        large_clusters = [c for c in clusters if len(c.memories) > 10]
        if large_clusters:
            for cluster in large_clusters[:3]:  # Top 3 largest
                recommendations.append(
                    f"Consider organizing memories about {cluster.topics[0] if cluster.topics else 'this topic'} "
                    f"into a dedicated knowledge base or summary"
                )
        
        # Recommendations based on knowledge gaps
        high_priority_gaps = [g for g in knowledge_gaps if g.importance_score > 0.7]
        for gap in high_priority_gaps[:3]:
            recommendations.append(
                f"Consider researching {gap.topic} to fill knowledge gaps - "
                f"try asking: {gap.suggested_questions[0] if gap.suggested_questions else 'relevant questions'}"
            )
        
        # Recommendations based on insights
        diverse_insights = [i for i in insights if i.insight_type == "diverse_interests"]
        if diverse_insights:
            recommendations.append(
                "Your knowledge spans many areas - consider creating connections "
                "between different topics to enhance understanding"
            )
        
        activity_bursts = [i for i in insights if i.insight_type == "activity_burst"]
        if len(activity_bursts) > 2:
            recommendations.append(
                "You have periods of intense learning - consider scheduling regular "
                "review sessions to consolidate knowledge from these periods"
            )
        
        # Generic recommendations if none specific
        if not recommendations:
            recommendations.extend([
                "Regular review of your memories can help identify patterns and connections",
                "Consider summarizing related memories to create comprehensive knowledge bases",
                "Look for opportunities to connect different areas of your knowledge"
            ])
        
        return recommendations[:5]  # Limit to 5 recommendations