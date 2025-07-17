"""
Local Organization Improvement Recommender.

This module provides intelligent recommendations for improving memory organization
using local clustering algorithms, structure analysis, impact prediction,
and tracking systems for optimal memory management and retrieval efficiency.
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
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.decomposition import PCA, LatentDirichletAllocation
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import networkx as nx
from collections import defaultdict, Counter
import spacy

import structlog
from pydantic import BaseModel, Field, ConfigDict, field_validator

from ...core.embeddings.embedder import Embedder
from ...core.memory.manager import MemoryManager
from ...core.graph.neo4j_client import Neo4jClient
from ...core.utils.config import settings

logger = structlog.get_logger(__name__)


class RecommendationType(str, Enum):
    """Types of organization improvement recommendations."""
    CLUSTERING_OPTIMIZATION = "clustering_optimization"
    TAG_RESTRUCTURING = "tag_restructuring"
    HIERARCHY_CREATION = "hierarchy_creation"
    DUPLICATE_CONSOLIDATION = "duplicate_consolidation"
    TOPIC_ORGANIZATION = "topic_organization"
    TEMPORAL_GROUPING = "temporal_grouping"
    ACCESS_PATTERN_OPTIMIZATION = "access_pattern_optimization"
    METADATA_ENHANCEMENT = "metadata_enhancement"
    RELATIONSHIP_STRENGTHENING = "relationship_strengthening"
    ARCHIVAL_SUGGESTION = "archival_suggestion"


class Priority(str, Enum):
    """Priority levels for recommendations."""
    CRITICAL = "critical"      # Must address immediately
    HIGH = "high"             # Address soon
    MEDIUM = "medium"         # Address when convenient
    LOW = "low"               # Address if time permits
    OPTIONAL = "optional"     # Nice to have


class ImpactLevel(str, Enum):
    """Expected impact levels of recommendations."""
    TRANSFORMATIVE = "transformative"  # Major improvement
    SIGNIFICANT = "significant"        # Notable improvement
    MODERATE = "moderate"              # Some improvement
    MINIMAL = "minimal"                # Small improvement
    UNCERTAIN = "uncertain"            # Impact unclear


@dataclass
class OrganizationContext:
    """Context for organization analysis and recommendations."""
    user_id: str
    analysis_scope: str = "all"  # "all", "recent", "frequent", "tagged"
    time_window_days: int = 30
    min_cluster_size: int = 3
    max_recommendations: int = 20
    include_low_priority: bool = False
    focus_areas: Optional[List[str]] = None  # Specific areas to focus on
    timestamp: datetime = field(default_factory=datetime.utcnow)


class OrganizationRecommendation(BaseModel):
    """Structured organization improvement recommendation."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    recommendation_id: str = Field(..., description="Unique recommendation identifier")
    recommendation_type: RecommendationType = Field(..., description="Type of recommendation")
    title: str = Field(..., min_length=5, description="Clear recommendation title")
    description: str = Field(..., min_length=20, description="Detailed description")
    
    # Priority and impact
    priority: Priority = Field(..., description="Recommendation priority")
    impact_level: ImpactLevel = Field(..., description="Expected impact level")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in recommendation")
    
    # Implementation details
    affected_memories: List[str] = Field(..., description="Memory IDs affected by recommendation")
    implementation_steps: List[str] = Field(..., description="Steps to implement recommendation")
    estimated_effort: str = Field(..., description="Estimated effort to implement")
    expected_benefits: List[str] = Field(..., description="Expected benefits")
    
    # Analysis data
    current_metrics: Dict[str, float] = Field(..., description="Current organization metrics")
    projected_metrics: Dict[str, float] = Field(..., description="Projected metrics after implementation")
    supporting_evidence: List[str] = Field(..., description="Evidence supporting recommendation")
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    dependencies: List[str] = Field(default_factory=list, description="Dependencies on other recommendations")
    tags: Set[str] = Field(default_factory=set, description="Recommendation tags")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class OrganizationAnalysis(BaseModel):
    """Comprehensive organization analysis results."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    recommendations: List[OrganizationRecommendation] = Field(..., description="Generated recommendations")
    total_memories_analyzed: int = Field(..., ge=0, description="Total memories analyzed")
    processing_time_ms: float = Field(..., ge=0.0, description="Analysis processing time")
    analysis_strategies: List[str] = Field(..., description="Analysis strategies used")
    
    # Current organization metrics
    organization_quality: Dict[str, float] = Field(..., description="Current organization quality metrics")
    cluster_analysis: Dict[str, Any] = Field(..., description="Clustering analysis results")
    structure_analysis: Dict[str, Any] = Field(..., description="Memory structure analysis")
    access_pattern_analysis: Dict[str, Any] = Field(..., description="Access pattern analysis")
    
    # Improvement potential
    improvement_potential: Dict[str, float] = Field(..., description="Potential improvement areas")
    priority_distribution: Dict[str, int] = Field(..., description="Distribution of recommendation priorities")
    
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Analysis timestamp")


class LocalOrganizationRecommender:
    """
    Local Organization Improvement Recommender.
    
    Analyzes memory organization and provides intelligent recommendations
    for improving structure, accessibility, and maintenance using local algorithms.
    """
    
    def __init__(
        self,
        embedder: Embedder,
        memory_manager: MemoryManager,
        graph_client: Optional[Neo4jClient] = None
    ):
        """Initialize local organization recommender."""
        self.embedder = embedder
        self.memory_manager = memory_manager
        self.graph_client = graph_client
        
        # Analysis components
        self.scaler = StandardScaler()
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.lda_model = LatentDirichletAllocation(n_components=10, random_state=42)
        
        # NLP processing
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found, using basic NLP")
            self.nlp = None
        
        # Caching
        self.analysis_cache: Dict[str, Dict[str, Any]] = {}
        self.embedding_cache: Dict[str, np.ndarray] = {}
        self.cluster_cache: Dict[str, Dict[str, Any]] = {}
        
        # Performance tracking
        self.recommendation_stats = {
            'total_analyses': 0,
            'total_recommendations': 0,
            'implementation_feedback': [],
            'average_processing_time': 0.0,
            'recommendation_by_type': {rtype.value: 0 for rtype in RecommendationType},
            'priority_distribution': {priority.value: 0 for priority in Priority}
        }
        
        logger.info("Initialized LocalOrganizationRecommender")
    
    async def analyze_organization(
        self,
        context: OrganizationContext
    ) -> OrganizationAnalysis:
        """
        Analyze memory organization and generate improvement recommendations.
        
        Args:
            context: Analysis context and parameters
            
        Returns:
            Comprehensive organization analysis with recommendations
        """
        start_time = datetime.utcnow()
        self.recommendation_stats['total_analyses'] += 1
        
        try:
            # Get memories for analysis
            memories = await self._get_memories_for_analysis(context)
            
            if len(memories) < context.min_cluster_size:
                return OrganizationAnalysis(
                    recommendations=[],
                    total_memories_analyzed=len(memories),
                    processing_time_ms=0.0,
                    analysis_strategies=[],
                    organization_quality={},
                    cluster_analysis={"status": "insufficient_memories"},
                    structure_analysis={},
                    access_pattern_analysis={},
                    improvement_potential={},
                    priority_distribution={}
                )
            
            # Perform multiple analysis strategies
            recommendations = []
            strategies_used = []
            
            # Strategy 1: Clustering analysis
            cluster_recommendations = await self._analyze_clustering(memories, context)
            recommendations.extend(cluster_recommendations)
            if cluster_recommendations:
                strategies_used.append("clustering_analysis")
            
            # Strategy 2: Tag structure analysis
            tag_recommendations = await self._analyze_tag_structure(memories, context)
            recommendations.extend(tag_recommendations)
            if tag_recommendations:
                strategies_used.append("tag_analysis")
            
            # Strategy 3: Topic organization analysis
            topic_recommendations = await self._analyze_topic_organization(memories, context)
            recommendations.extend(topic_recommendations)
            if topic_recommendations:
                strategies_used.append("topic_analysis")
            
            # Strategy 4: Temporal organization analysis
            temporal_recommendations = await self._analyze_temporal_organization(memories, context)
            recommendations.extend(temporal_recommendations)
            if temporal_recommendations:
                strategies_used.append("temporal_analysis")
            
            # Strategy 5: Access pattern analysis
            access_recommendations = await self._analyze_access_patterns(memories, context)
            recommendations.extend(access_recommendations)
            if access_recommendations:
                strategies_used.append("access_pattern_analysis")
            
            # Strategy 6: Duplicate detection
            duplicate_recommendations = await self._analyze_duplicates(memories, context)
            recommendations.extend(duplicate_recommendations)
            if duplicate_recommendations:
                strategies_used.append("duplicate_analysis")
            
            # Strategy 7: Metadata enhancement
            metadata_recommendations = await self._analyze_metadata_quality(memories, context)
            recommendations.extend(metadata_recommendations)
            if metadata_recommendations:
                strategies_used.append("metadata_analysis")
            
            # Strategy 8: Relationship analysis
            relationship_recommendations = await self._analyze_relationships(memories, context)
            recommendations.extend(relationship_recommendations)
            if relationship_recommendations:
                strategies_used.append("relationship_analysis")
            
            # Remove duplicates and prioritize
            unique_recommendations = self._deduplicate_recommendations(recommendations)
            prioritized_recommendations = self._prioritize_recommendations(
                unique_recommendations, context
            )
            
            # Limit to max recommendations
            final_recommendations = prioritized_recommendations[:context.max_recommendations]
            
            # Calculate comprehensive metrics
            organization_quality = await self._calculate_organization_quality(memories)
            cluster_analysis = await self._perform_cluster_analysis(memories)
            structure_analysis = await self._analyze_memory_structure(memories)
            access_pattern_analysis = await self._analyze_access_patterns_detailed(memories, context)
            improvement_potential = self._calculate_improvement_potential(final_recommendations)
            
            # Priority distribution
            priority_distribution = Counter(r.priority for r in final_recommendations)
            
            # Update statistics
            for rec in final_recommendations:
                self.recommendation_stats['recommendation_by_type'][rec.recommendation_type.value] += 1
                self.recommendation_stats['priority_distribution'][rec.priority.value] += 1
            
            self.recommendation_stats['total_recommendations'] += len(final_recommendations)
            
            # Calculate processing time
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            self.recommendation_stats['average_processing_time'] = (
                (self.recommendation_stats['average_processing_time'] * 
                 (self.recommendation_stats['total_analyses'] - 1) + processing_time) /
                self.recommendation_stats['total_analyses']
            )
            
            logger.info(
                "Generated organization recommendations",
                user_id=context.user_id,
                recommendations_count=len(final_recommendations),
                memories_analyzed=len(memories),
                processing_time_ms=processing_time,
                strategies=strategies_used
            )
            
            return OrganizationAnalysis(
                recommendations=final_recommendations,
                total_memories_analyzed=len(memories),
                processing_time_ms=processing_time,
                analysis_strategies=strategies_used,
                organization_quality=organization_quality,
                cluster_analysis=cluster_analysis,
                structure_analysis=structure_analysis,
                access_pattern_analysis=access_pattern_analysis,
                improvement_potential=improvement_potential,
                priority_distribution=dict(priority_distribution)
            )
            
        except Exception as e:
            logger.error("Error analyzing organization", error=str(e))
            return OrganizationAnalysis(
                recommendations=[],
                total_memories_analyzed=0,
                processing_time_ms=0.0,
                analysis_strategies=[],
                organization_quality={},
                cluster_analysis={"error": str(e)},
                structure_analysis={},
                access_pattern_analysis={},
                improvement_potential={},
                priority_distribution={}
            )
    
    async def _get_memories_for_analysis(self, context: OrganizationContext) -> List[Dict[str, Any]]:
        """Get memories for organization analysis based on context."""
        try:
            if context.analysis_scope == "recent":
                # Get recent memories
                cutoff_date = datetime.utcnow() - timedelta(days=context.time_window_days)
                memories = await self.memory_manager.get_memories_for_user(
                    context.user_id,
                    created_after=cutoff_date,
                    limit=500
                )
            else:
                # Get all memories
                memories = await self.memory_manager.get_memories_for_user(
                    context.user_id,
                    limit=500
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
                    "entities": getattr(memory, 'entities', []),
                    "access_count": getattr(memory, 'access_count', 0),
                    "last_accessed": getattr(memory, 'last_accessed', memory.created_at)
                }
                memory_list.append(memory_dict)
            
            logger.debug(f"Retrieved {len(memory_list)} memories for organization analysis")
            return memory_list
            
        except Exception as e:
            logger.error("Error retrieving memories for analysis", error=str(e))
            return []
    
    async def _analyze_clustering(
        self,
        memories: List[Dict[str, Any]],
        context: OrganizationContext
    ) -> List[OrganizationRecommendation]:
        """Analyze clustering opportunities and recommend improvements."""
        recommendations = []
        
        try:
            if len(memories) < context.min_cluster_size * 2:
                return recommendations
            
            # Get embeddings for all memories
            embeddings = []
            memory_ids = []
            
            for memory in memories:
                embedding = await self._get_or_compute_embedding(memory["content"])
                if embedding is not None:
                    embeddings.append(embedding)
                    memory_ids.append(memory["id"])
            
            if len(embeddings) < context.min_cluster_size * 2:
                return recommendations
            
            embeddings_matrix = np.array(embeddings)
            
            # Try different clustering algorithms
            clustering_results = {}
            
            # K-means clustering
            for n_clusters in [3, 5, 8, 10]:
                if n_clusters < len(embeddings):
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                    cluster_labels = kmeans.fit_predict(embeddings_matrix)
                    
                    # Calculate silhouette score
                    if len(np.unique(cluster_labels)) > 1:
                        sil_score = silhouette_score(embeddings_matrix, cluster_labels)
                        clustering_results[f"kmeans_{n_clusters}"] = {
                            "labels": cluster_labels,
                            "score": sil_score,
                            "n_clusters": n_clusters,
                            "algorithm": "kmeans"
                        }
            
            # DBSCAN clustering
            for eps in [0.3, 0.5, 0.7]:
                dbscan = DBSCAN(eps=eps, min_samples=context.min_cluster_size)
                cluster_labels = dbscan.fit_predict(embeddings_matrix)
                
                n_clusters = len(np.unique(cluster_labels)) - (1 if -1 in cluster_labels else 0)
                if n_clusters > 1:
                    sil_score = silhouette_score(embeddings_matrix, cluster_labels)
                    clustering_results[f"dbscan_{eps}"] = {
                        "labels": cluster_labels,
                        "score": sil_score,
                        "n_clusters": n_clusters,
                        "algorithm": "dbscan",
                        "eps": eps
                    }
            
            # Find best clustering
            if clustering_results:
                best_clustering = max(clustering_results.values(), key=lambda x: x["score"])
                
                if best_clustering["score"] > 0.3:  # Decent clustering quality
                    # Analyze cluster quality
                    cluster_labels = best_clustering["labels"]
                    cluster_analysis = self._analyze_cluster_quality(
                        memories, memory_ids, cluster_labels, embeddings_matrix
                    )
                    
                    # Generate clustering recommendation
                    rec = self._create_clustering_recommendation(
                        cluster_analysis, best_clustering, memory_ids, context
                    )
                    recommendations.append(rec)
            
        except Exception as e:
            logger.error("Error in clustering analysis", error=str(e))
        
        return recommendations
    
    def _analyze_cluster_quality(
        self,
        memories: List[Dict[str, Any]],
        memory_ids: List[str],
        cluster_labels: np.ndarray,
        embeddings_matrix: np.ndarray
    ) -> Dict[str, Any]:
        """Analyze the quality of clusters."""
        cluster_info = defaultdict(list)
        
        # Group memories by cluster
        for i, label in enumerate(cluster_labels):
            if label != -1:  # Ignore noise points in DBSCAN
                memory_id = memory_ids[i]
                memory = next(m for m in memories if m["id"] == memory_id)
                cluster_info[label].append({
                    "id": memory_id,
                    "content": memory["content"][:200],  # First 200 chars
                    "tags": memory["tags"],
                    "created_at": memory["created_at"]
                })
        
        # Analyze cluster coherence
        cluster_analysis = {
            "clusters": {},
            "total_clusters": len(cluster_info),
            "avg_cluster_size": np.mean([len(cluster) for cluster in cluster_info.values()]),
            "cluster_size_std": np.std([len(cluster) for cluster in cluster_info.values()]),
            "coherence_scores": {}
        }
        
        for cluster_id, cluster_memories in cluster_info.items():
            # Calculate intra-cluster coherence
            cluster_indices = [i for i, label in enumerate(cluster_labels) if label == cluster_id]
            if len(cluster_indices) > 1:
                cluster_embeddings = embeddings_matrix[cluster_indices]
                pairwise_similarities = []
                
                for i in range(len(cluster_embeddings)):
                    for j in range(i + 1, len(cluster_embeddings)):
                        sim = np.dot(cluster_embeddings[i], cluster_embeddings[j]) / (
                            np.linalg.norm(cluster_embeddings[i]) * np.linalg.norm(cluster_embeddings[j])
                        )
                        pairwise_similarities.append(sim)
                
                coherence = np.mean(pairwise_similarities) if pairwise_similarities else 0
                cluster_analysis["coherence_scores"][cluster_id] = coherence
            
            # Extract common topics/themes
            cluster_texts = [mem["content"] for mem in cluster_memories]
            common_words = self._extract_common_themes(cluster_texts)
            
            cluster_analysis["clusters"][cluster_id] = {
                "size": len(cluster_memories),
                "common_themes": common_words[:5],
                "memory_ids": [mem["id"] for mem in cluster_memories],
                "time_span": self._calculate_time_span(cluster_memories),
                "coherence": cluster_analysis["coherence_scores"].get(cluster_id, 0)
            }
        
        return cluster_analysis
    
    def _create_clustering_recommendation(
        self,
        cluster_analysis: Dict[str, Any],
        best_clustering: Dict[str, Any],
        memory_ids: List[str],
        context: OrganizationContext
    ) -> OrganizationRecommendation:
        """Create clustering optimization recommendation."""
        rec_id = hashlib.md5(f"clustering_{context.user_id}_{context.timestamp}".encode()).hexdigest()[:16]
        
        # Determine recommendation details based on cluster quality
        avg_coherence = np.mean(list(cluster_analysis["coherence_scores"].values()))
        cluster_count = cluster_analysis["total_clusters"]
        
        if avg_coherence > 0.7:
            priority = Priority.LOW
            title = "Optimize Existing Memory Clusters"
            description = f"Your memories are well-clustered into {cluster_count} groups with good coherence. Consider adding cluster-based tags for easier navigation."
            impact = ImpactLevel.MODERATE
        elif avg_coherence > 0.5:
            priority = Priority.MEDIUM
            title = "Improve Memory Clustering Organization"
            description = f"Your memories show {cluster_count} natural clusters that could be better organized. Implementing cluster-based organization would improve findability."
            impact = ImpactLevel.SIGNIFICANT
        else:
            priority = Priority.HIGH
            title = "Reorganize Memories into Coherent Groups"
            description = f"Memory analysis reveals {cluster_count} potential groups, but they lack clear organization. Restructuring would significantly improve accessibility."
            impact = ImpactLevel.TRANSFORMATIVE
        
        # Generate implementation steps
        implementation_steps = [
            f"Review the {cluster_count} identified memory clusters",
            "Create descriptive tags for each cluster theme",
            "Apply cluster-based tags to organize memories",
            "Consider creating cluster-based folders or categories",
            "Set up cluster-based search filters"
        ]
        
        # Calculate projected improvements
        current_metrics = {
            "cluster_coherence": avg_coherence,
            "organization_score": avg_coherence * 0.8,
            "findability_score": min(avg_coherence + 0.2, 1.0)
        }
        
        projected_metrics = {
            "cluster_coherence": min(avg_coherence + 0.3, 1.0),
            "organization_score": min(avg_coherence + 0.4, 1.0),
            "findability_score": min(avg_coherence + 0.5, 1.0)
        }
        
        # Evidence supporting recommendation
        evidence = [
            f"Identified {cluster_count} natural clusters in your memories",
            f"Average cluster coherence: {avg_coherence:.2f}",
            f"Clustering algorithm: {best_clustering['algorithm']}",
            f"Silhouette score: {best_clustering['score']:.3f}"
        ]
        
        # Benefits
        benefits = [
            "Improved memory findability through logical grouping",
            "Faster retrieval of related memories",
            "Better understanding of memory content themes",
            "Enhanced browsing and exploration experience"
        ]
        
        return OrganizationRecommendation(
            recommendation_id=rec_id,
            recommendation_type=RecommendationType.CLUSTERING_OPTIMIZATION,
            title=title,
            description=description,
            priority=priority,
            impact_level=impact,
            confidence=min(best_clustering["score"] + 0.2, 1.0),
            affected_memories=memory_ids,
            implementation_steps=implementation_steps,
            estimated_effort="2-4 hours",
            expected_benefits=benefits,
            current_metrics=current_metrics,
            projected_metrics=projected_metrics,
            supporting_evidence=evidence,
            tags={"clustering", "organization", "findability"},
            metadata={"cluster_analysis": cluster_analysis, "clustering_params": best_clustering}
        )
    
    async def _analyze_tag_structure(
        self,
        memories: List[Dict[str, Any]],
        context: OrganizationContext
    ) -> List[OrganizationRecommendation]:
        """Analyze tag structure and recommend improvements."""
        recommendations = []
        
        try:
            # Collect all tags
            all_tags = []
            tag_usage = Counter()
            memories_by_tag = defaultdict(list)
            untagged_memories = []
            
            for memory in memories:
                memory_tags = memory.get("tags", [])
                if memory_tags:
                    all_tags.extend(memory_tags)
                    for tag in memory_tags:
                        tag_usage[tag] += 1
                        memories_by_tag[tag].append(memory["id"])
                else:
                    untagged_memories.append(memory["id"])
            
            # Analyze tag quality
            tag_analysis = {
                "total_tags": len(set(all_tags)),
                "total_tag_uses": len(all_tags),
                "untagged_count": len(untagged_memories),
                "untagged_ratio": len(untagged_memories) / len(memories),
                "avg_tags_per_memory": len(all_tags) / max(len(memories), 1),
                "most_used_tags": tag_usage.most_common(10),
                "rarely_used_tags": [tag for tag, count in tag_usage.items() if count == 1],
                "tag_distribution": dict(tag_usage)
            }
            
            # Generate tag-related recommendations
            
            # 1. Untagged memories recommendation
            if tag_analysis["untagged_ratio"] > 0.3:
                rec = self._create_tag_enhancement_recommendation(
                    tag_analysis, untagged_memories, context
                )
                recommendations.append(rec)
            
            # 2. Tag consolidation recommendation
            if len(tag_analysis["rarely_used_tags"]) > len(set(all_tags)) * 0.3:
                rec = self._create_tag_consolidation_recommendation(
                    tag_analysis, context
                )
                recommendations.append(rec)
            
            # 3. Tag hierarchy recommendation
            if tag_analysis["total_tags"] > 20:
                rec = self._create_tag_hierarchy_recommendation(
                    tag_analysis, memories_by_tag, context
                )
                recommendations.append(rec)
            
        except Exception as e:
            logger.error("Error in tag structure analysis", error=str(e))
        
        return recommendations
    
    def _create_tag_enhancement_recommendation(
        self,
        tag_analysis: Dict[str, Any],
        untagged_memories: List[str],
        context: OrganizationContext
    ) -> OrganizationRecommendation:
        """Create recommendation for enhancing tag coverage."""
        rec_id = hashlib.md5(f"tags_{context.user_id}_{len(untagged_memories)}".encode()).hexdigest()[:16]
        
        untagged_ratio = tag_analysis["untagged_ratio"]
        
        if untagged_ratio > 0.7:
            priority = Priority.HIGH
            impact = ImpactLevel.TRANSFORMATIVE
            title = "Add Tags to Improve Memory Organization"
            description = f"{len(untagged_memories)} memories ({untagged_ratio:.1%}) lack tags, making them hard to find and organize."
        elif untagged_ratio > 0.5:
            priority = Priority.MEDIUM
            impact = ImpactLevel.SIGNIFICANT
            title = "Enhance Memory Tagging for Better Organization"
            description = f"{len(untagged_memories)} memories ({untagged_ratio:.1%}) need tags to improve discoverability."
        else:
            priority = Priority.LOW
            impact = ImpactLevel.MODERATE
            title = "Complete Memory Tagging for Full Organization"
            description = f"{len(untagged_memories)} memories ({untagged_ratio:.1%}) could benefit from tags."
        
        implementation_steps = [
            "Review untagged memories to identify common themes",
            "Create a consistent tagging vocabulary",
            "Apply relevant tags to untagged memories",
            "Establish tagging guidelines for future memories",
            "Set up automated tagging suggestions if possible"
        ]
        
        current_metrics = {
            "tagged_ratio": 1 - untagged_ratio,
            "avg_tags_per_memory": tag_analysis["avg_tags_per_memory"],
            "findability_score": (1 - untagged_ratio) * 0.8
        }
        
        projected_metrics = {
            "tagged_ratio": 0.95,
            "avg_tags_per_memory": max(tag_analysis["avg_tags_per_memory"], 2.0),
            "findability_score": 0.9
        }
        
        evidence = [
            f"{len(untagged_memories)} memories without tags",
            f"Current tagging rate: {(1-untagged_ratio):.1%}",
            f"Average tags per memory: {tag_analysis['avg_tags_per_memory']:.1f}"
        ]
        
        benefits = [
            "Dramatically improved memory searchability",
            "Better categorization and browsing",
            "Easier identification of related memories",
            "Enhanced memory maintenance and cleanup"
        ]
        
        return OrganizationRecommendation(
            recommendation_id=rec_id,
            recommendation_type=RecommendationType.TAG_RESTRUCTURING,
            title=title,
            description=description,
            priority=priority,
            impact_level=impact,
            confidence=0.9,  # High confidence in tagging benefits
            affected_memories=untagged_memories,
            implementation_steps=implementation_steps,
            estimated_effort="3-6 hours",
            expected_benefits=benefits,
            current_metrics=current_metrics,
            projected_metrics=projected_metrics,
            supporting_evidence=evidence,
            tags={"tags", "organization", "searchability"}
        )
    
    def _create_tag_consolidation_recommendation(
        self,
        tag_analysis: Dict[str, Any],
        context: OrganizationContext
    ) -> OrganizationRecommendation:
        """Create recommendation for consolidating rarely used tags."""
        rec_id = hashlib.md5(f"tag_consolidation_{context.user_id}".encode()).hexdigest()[:16]
        
        rarely_used_count = len(tag_analysis["rarely_used_tags"])
        total_tags = tag_analysis["total_tags"]
        
        priority = Priority.MEDIUM if rarely_used_count > total_tags * 0.5 else Priority.LOW
        impact = ImpactLevel.MODERATE
        
        title = "Consolidate Rarely Used Tags"
        description = f"{rarely_used_count} tags are used only once. Consolidating similar tags would reduce clutter and improve consistency."
        
        implementation_steps = [
            "Review rarely used tags for consolidation opportunities",
            "Identify similar or redundant tags",
            "Create a unified tagging vocabulary",
            "Merge or rename similar tags",
            "Update affected memories with consolidated tags"
        ]
        
        # Find affected memories
        affected_memories = []
        for tag in tag_analysis["rarely_used_tags"]:
            # This would require access to memories_by_tag from the calling function
            # For now, we'll estimate
            affected_memories.extend([f"memory_with_{tag}"])  # Placeholder
        
        current_metrics = {
            "tag_efficiency": (total_tags - rarely_used_count) / total_tags,
            "tag_consistency": 0.6,  # Estimated
            "maintenance_burden": rarely_used_count / total_tags
        }
        
        projected_metrics = {
            "tag_efficiency": min((total_tags - rarely_used_count * 0.7) / total_tags, 1.0),
            "tag_consistency": 0.8,
            "maintenance_burden": (rarely_used_count * 0.3) / total_tags
        }
        
        evidence = [
            f"{rarely_used_count} tags used only once",
            f"{rarely_used_count/total_tags:.1%} of tags are rarely used",
            "Tag consolidation reduces maintenance overhead"
        ]
        
        benefits = [
            "Cleaner, more consistent tagging system",
            "Reduced cognitive load when selecting tags",
            "Easier tag maintenance and evolution",
            "Better tag-based search results"
        ]
        
        return OrganizationRecommendation(
            recommendation_id=rec_id,
            recommendation_type=RecommendationType.TAG_RESTRUCTURING,
            title=title,
            description=description,
            priority=priority,
            impact_level=impact,
            confidence=0.8,
            affected_memories=affected_memories[:20],  # Limit for display
            implementation_steps=implementation_steps,
            estimated_effort="2-3 hours",
            expected_benefits=benefits,
            current_metrics=current_metrics,
            projected_metrics=projected_metrics,
            supporting_evidence=evidence,
            tags={"tags", "consolidation", "maintenance"}
        )
    
    def _create_tag_hierarchy_recommendation(
        self,
        tag_analysis: Dict[str, Any],
        memories_by_tag: Dict[str, List[str]],
        context: OrganizationContext
    ) -> OrganizationRecommendation:
        """Create recommendation for establishing tag hierarchy."""
        rec_id = hashlib.md5(f"tag_hierarchy_{context.user_id}".encode()).hexdigest()[:16]
        
        total_tags = tag_analysis["total_tags"]
        
        title = "Create Tag Hierarchy for Better Organization"
        description = f"With {total_tags} tags, a hierarchical structure would improve navigation and reduce tag selection complexity."
        
        priority = Priority.MEDIUM
        impact = ImpactLevel.SIGNIFICANT
        
        # Analyze tag relationships (simplified)
        tag_relationships = self._analyze_tag_relationships(memories_by_tag)
        
        implementation_steps = [
            "Group related tags into categories",
            "Create parent-child tag relationships",
            "Establish tag naming conventions",
            "Implement hierarchical tag display",
            "Train users on the new tag structure"
        ]
        
        current_metrics = {
            "tag_organization": 0.4,  # Estimated flat organization
            "tag_findability": 0.5,
            "user_efficiency": 0.6
        }
        
        projected_metrics = {
            "tag_organization": 0.8,
            "tag_findability": 0.9,
            "user_efficiency": 0.85
        }
        
        evidence = [
            f"{total_tags} tags suggest need for hierarchy",
            "Flat tag structure becomes unwieldy at scale",
            f"Identified {len(tag_relationships)} potential tag groupings"
        ]
        
        benefits = [
            "Improved tag browsing and selection",
            "Better understanding of tag relationships",
            "Reduced tag redundancy and conflicts",
            "Enhanced memory categorization"
        ]
        
        # Affected memories are those with the most common tags
        affected_memories = []
        for tag, memory_ids in memories_by_tag.items():
            if len(memory_ids) > 3:  # Only include frequently used tags
                affected_memories.extend(memory_ids[:5])  # Limit per tag
        
        return OrganizationRecommendation(
            recommendation_id=rec_id,
            recommendation_type=RecommendationType.HIERARCHY_CREATION,
            title=title,
            description=description,
            priority=priority,
            impact_level=impact,
            confidence=0.7,
            affected_memories=list(set(affected_memories))[:30],
            implementation_steps=implementation_steps,
            estimated_effort="4-8 hours",
            expected_benefits=benefits,
            current_metrics=current_metrics,
            projected_metrics=projected_metrics,
            supporting_evidence=evidence,
            tags={"hierarchy", "tags", "structure", "navigation"}
        )
    
    def _analyze_tag_relationships(self, memories_by_tag: Dict[str, List[str]]) -> Dict[str, Set[str]]:
        """Analyze relationships between tags based on co-occurrence."""
        tag_relationships = defaultdict(set)
        
        # Find tags that frequently co-occur
        for tag1, memories1 in memories_by_tag.items():
            for tag2, memories2 in memories_by_tag.items():
                if tag1 != tag2:
                    overlap = len(set(memories1).intersection(set(memories2)))
                    if overlap > 2:  # Threshold for relationship
                        tag_relationships[tag1].add(tag2)
        
        return dict(tag_relationships)
    
    async def _analyze_topic_organization(
        self,
        memories: List[Dict[str, Any]],
        context: OrganizationContext
    ) -> List[OrganizationRecommendation]:
        """Analyze topic organization and recommend improvements."""
        recommendations = []
        
        try:
            if len(memories) < 10:
                return recommendations
            
            # Extract text content
            texts = [memory["content"] for memory in memories]
            
            # Perform topic modeling
            try:
                tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
                topic_distributions = self.lda_model.fit_transform(tfidf_matrix)
                
                # Analyze topic coherence and distribution
                topic_analysis = self._analyze_topic_coherence(
                    topic_distributions, texts, memories
                )
                
                # Generate topic-based recommendations
                if topic_analysis["coherence_score"] < 0.6:
                    rec = self._create_topic_organization_recommendation(
                        topic_analysis, memories, context
                    )
                    recommendations.append(rec)
                    
            except Exception as e:
                logger.warning(f"Topic modeling failed: {e}")
                
        except Exception as e:
            logger.error("Error in topic organization analysis", error=str(e))
        
        return recommendations
    
    def _analyze_topic_coherence(
        self,
        topic_distributions: np.ndarray,
        texts: List[str],
        memories: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze coherence of topic organization."""
        n_topics = topic_distributions.shape[1]
        
        # Assign each memory to its dominant topic
        dominant_topics = np.argmax(topic_distributions, axis=1)
        topic_assignments = defaultdict(list)
        
        for i, topic in enumerate(dominant_topics):
            topic_assignments[topic].append({
                "memory_id": memories[i]["id"],
                "content": texts[i][:200],
                "topic_strength": topic_distributions[i][topic]
            })
        
        # Calculate topic coherence
        topic_coherences = []
        for topic_id, memories_in_topic in topic_assignments.items():
            if len(memories_in_topic) > 1:
                # Simple coherence based on topic strength
                strengths = [mem["topic_strength"] for mem in memories_in_topic]
                coherence = np.mean(strengths)
                topic_coherences.append(coherence)
        
        overall_coherence = np.mean(topic_coherences) if topic_coherences else 0
        
        return {
            "n_topics": n_topics,
            "topic_assignments": dict(topic_assignments),
            "coherence_score": overall_coherence,
            "topic_sizes": {k: len(v) for k, v in topic_assignments.items()},
            "orphaned_topics": [k for k, v in topic_assignments.items() if len(v) == 1]
        }
    
    def _create_topic_organization_recommendation(
        self,
        topic_analysis: Dict[str, Any],
        memories: List[Dict[str, Any]],
        context: OrganizationContext
    ) -> OrganizationRecommendation:
        """Create topic organization recommendation."""
        rec_id = hashlib.md5(f"topics_{context.user_id}_{topic_analysis['n_topics']}".encode()).hexdigest()[:16]
        
        coherence = topic_analysis["coherence_score"]
        n_topics = topic_analysis["n_topics"]
        
        if coherence < 0.4:
            priority = Priority.HIGH
            impact = ImpactLevel.SIGNIFICANT
            title = "Reorganize Memories by Topics for Better Structure"
            description = f"Topic analysis reveals {n_topics} themes with low coherence ({coherence:.2f}). Reorganizing by topics would improve accessibility."
        else:
            priority = Priority.MEDIUM
            impact = ImpactLevel.MODERATE
            title = "Enhance Topic-Based Organization"
            description = f"Your memories show {n_topics} topics that could be better organized for improved navigation."
        
        # Get affected memories (all memories in this case)
        affected_memories = [memory["id"] for memory in memories]
        
        implementation_steps = [
            "Review identified topic themes",
            "Create topic-based categories or tags",
            "Reorganize memories into topic groups",
            "Establish topic naming conventions",
            "Set up topic-based navigation"
        ]
        
        current_metrics = {
            "topic_coherence": coherence,
            "topic_organization": coherence * 0.8,
            "content_findability": coherence * 0.9
        }
        
        projected_metrics = {
            "topic_coherence": min(coherence + 0.3, 1.0),
            "topic_organization": min(coherence + 0.4, 1.0),
            "content_findability": min(coherence + 0.35, 1.0)
        }
        
        evidence = [
            f"Identified {n_topics} distinct topics",
            f"Current topic coherence: {coherence:.2f}",
            f"Topic size distribution: {list(topic_analysis['topic_sizes'].values())}"
        ]
        
        benefits = [
            "Better content organization by theme",
            "Improved topic-based browsing",
            "Enhanced content discovery",
            "Clearer content categorization"
        ]
        
        return OrganizationRecommendation(
            recommendation_id=rec_id,
            recommendation_type=RecommendationType.TOPIC_ORGANIZATION,
            title=title,
            description=description,
            priority=priority,
            impact_level=impact,
            confidence=0.8,
            affected_memories=affected_memories,
            implementation_steps=implementation_steps,
            estimated_effort="3-5 hours",
            expected_benefits=benefits,
            current_metrics=current_metrics,
            projected_metrics=projected_metrics,
            supporting_evidence=evidence,
            tags={"topics", "organization", "themes"},
            metadata={"topic_analysis": topic_analysis}
        )
    
    async def _analyze_temporal_organization(
        self,
        memories: List[Dict[str, Any]],
        context: OrganizationContext
    ) -> List[OrganizationRecommendation]:
        """Analyze temporal organization patterns."""
        recommendations = []
        
        try:
            # Analyze temporal distribution
            dates = [memory["created_at"] for memory in memories]
            dates.sort()
            
            # Calculate temporal metrics
            time_span = (dates[-1] - dates[0]).days if len(dates) > 1 else 0
            
            # Group by time periods
            temporal_groups = self._group_memories_by_time(memories)
            
            # Analyze temporal organization quality
            temporal_analysis = {
                "time_span_days": time_span,
                "temporal_groups": temporal_groups,
                "distribution_quality": self._calculate_temporal_distribution_quality(temporal_groups),
                "access_pattern_alignment": self._analyze_temporal_access_patterns(memories)
            }
            
            # Generate recommendation if temporal organization could be improved
            if temporal_analysis["distribution_quality"] < 0.6 and time_span > 30:
                rec = self._create_temporal_organization_recommendation(
                    temporal_analysis, memories, context
                )
                recommendations.append(rec)
                
        except Exception as e:
            logger.error("Error in temporal organization analysis", error=str(e))
        
        return recommendations
    
    def _group_memories_by_time(self, memories: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Group memories by time periods."""
        groups = {
            "last_week": [],
            "last_month": [],
            "last_quarter": [],
            "last_year": [],
            "older": []
        }
        
        now = datetime.utcnow()
        
        for memory in memories:
            created_at = memory["created_at"]
            days_ago = (now - created_at).days
            
            if days_ago <= 7:
                groups["last_week"].append(memory["id"])
            elif days_ago <= 30:
                groups["last_month"].append(memory["id"])
            elif days_ago <= 90:
                groups["last_quarter"].append(memory["id"])
            elif days_ago <= 365:
                groups["last_year"].append(memory["id"])
            else:
                groups["older"].append(memory["id"])
        
        return groups
    
    def _calculate_temporal_distribution_quality(self, temporal_groups: Dict[str, List[str]]) -> float:
        """Calculate quality of temporal distribution."""
        # Simple heuristic: better distribution = more even spread across time periods
        group_sizes = [len(group) for group in temporal_groups.values()]
        total_memories = sum(group_sizes)
        
        if total_memories == 0:
            return 0.0
        
        # Calculate entropy-based distribution quality
        probabilities = [size / total_memories for size in group_sizes if size > 0]
        if len(probabilities) <= 1:
            return 0.5
        
        entropy = -sum(p * math.log2(p) for p in probabilities)
        max_entropy = math.log2(len(probabilities))
        
        return entropy / max_entropy if max_entropy > 0 else 0.0
    
    def _analyze_temporal_access_patterns(self, memories: List[Dict[str, Any]]) -> float:
        """Analyze how well temporal organization aligns with access patterns."""
        # Simplified analysis - in reality would use actual access data
        recent_access_score = 0.0
        total_accessed = 0
        
        now = datetime.utcnow()
        
        for memory in memories:
            last_accessed = memory.get("last_accessed", memory["created_at"])
            access_count = memory.get("access_count", 1)
            
            if access_count > 1:
                total_accessed += 1
                days_since_access = (now - last_accessed).days
                
                # Recent access gets higher score
                if days_since_access <= 7:
                    recent_access_score += 1.0
                elif days_since_access <= 30:
                    recent_access_score += 0.7
                elif days_since_access <= 90:
                    recent_access_score += 0.4
                else:
                    recent_access_score += 0.1
        
        return recent_access_score / max(total_accessed, 1)
    
    def _create_temporal_organization_recommendation(
        self,
        temporal_analysis: Dict[str, Any],
        memories: List[Dict[str, Any]],
        context: OrganizationContext
    ) -> OrganizationRecommendation:
        """Create temporal organization recommendation."""
        rec_id = hashlib.md5(f"temporal_{context.user_id}".encode()).hexdigest()[:16]
        
        distribution_quality = temporal_analysis["distribution_quality"]
        time_span = temporal_analysis["time_span_days"]
        
        if distribution_quality < 0.4:
            priority = Priority.MEDIUM
            impact = ImpactLevel.MODERATE
            title = "Organize Memories by Time Periods"
            description = f"With memories spanning {time_span} days, temporal organization would improve browsing and retrieval."
        else:
            priority = Priority.LOW
            impact = ImpactLevel.MINIMAL
            title = "Enhance Temporal Memory Navigation"
            description = f"Your {time_span}-day memory span could benefit from better temporal organization."
        
        implementation_steps = [
            "Create time-based folders or categories",
            "Implement date-based memory grouping",
            "Set up temporal navigation features",
            "Add time-based search filters",
            "Consider archiving very old memories"
        ]
        
        affected_memories = [memory["id"] for memory in memories]
        
        current_metrics = {
            "temporal_organization": distribution_quality,
            "browsing_efficiency": distribution_quality * 0.8,
            "archive_readiness": self._calculate_archive_readiness(memories)
        }
        
        projected_metrics = {
            "temporal_organization": min(distribution_quality + 0.3, 1.0),
            "browsing_efficiency": min(distribution_quality + 0.4, 1.0),
            "archive_readiness": 0.8
        }
        
        evidence = [
            f"Memory span: {time_span} days",
            f"Temporal distribution quality: {distribution_quality:.2f}",
            f"Temporal groups: {[len(g) for g in temporal_analysis['temporal_groups'].values()]}"
        ]
        
        benefits = [
            "Improved chronological browsing",
            "Better time-based memory retrieval",
            "Enhanced memory lifecycle management",
            "Clearer memory timeline navigation"
        ]
        
        return OrganizationRecommendation(
            recommendation_id=rec_id,
            recommendation_type=RecommendationType.TEMPORAL_GROUPING,
            title=title,
            description=description,
            priority=priority,
            impact_level=impact,
            confidence=0.7,
            affected_memories=affected_memories,
            implementation_steps=implementation_steps,
            estimated_effort="2-4 hours",
            expected_benefits=benefits,
            current_metrics=current_metrics,
            projected_metrics=projected_metrics,
            supporting_evidence=evidence,
            tags={"temporal", "organization", "chronological"}
        )
    
    def _calculate_archive_readiness(self, memories: List[Dict[str, Any]]) -> float:
        """Calculate how ready memories are for archival."""
        now = datetime.utcnow()
        archivable_count = 0
        
        for memory in memories:
            last_accessed = memory.get("last_accessed", memory["created_at"])
            days_since_access = (now - last_accessed).days
            access_count = memory.get("access_count", 1)
            
            # Memory is archivable if old and rarely accessed
            if days_since_access > 180 and access_count <= 2:
                archivable_count += 1
        
        return archivable_count / max(len(memories), 1)
    
    async def _analyze_access_patterns(
        self,
        memories: List[Dict[str, Any]],
        context: OrganizationContext
    ) -> List[OrganizationRecommendation]:
        """Analyze access patterns and recommend optimizations."""
        recommendations = []
        
        try:
            # Analyze access frequency and recency
            access_analysis = self._analyze_memory_access_patterns(memories)
            
            # Generate access pattern recommendations
            if access_analysis["optimization_potential"] > 0.3:
                rec = self._create_access_optimization_recommendation(
                    access_analysis, memories, context
                )
                recommendations.append(rec)
                
        except Exception as e:
            logger.error("Error in access pattern analysis", error=str(e))
        
        return recommendations
    
    def _analyze_memory_access_patterns(self, memories: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze memory access patterns."""
        now = datetime.utcnow()
        
        access_stats = {
            "frequently_accessed": [],
            "rarely_accessed": [],
            "never_accessed": [],
            "recently_accessed": [],
            "stale_memories": []
        }
        
        total_accesses = 0
        
        for memory in memories:
            access_count = memory.get("access_count", 1)
            last_accessed = memory.get("last_accessed", memory["created_at"])
            days_since_access = (now - last_accessed).days
            
            total_accesses += access_count
            
            # Categorize by access frequency
            if access_count >= 5:
                access_stats["frequently_accessed"].append(memory["id"])
            elif access_count <= 1:
                access_stats["never_accessed"].append(memory["id"])
            else:
                access_stats["rarely_accessed"].append(memory["id"])
            
            # Categorize by recency
            if days_since_access <= 7:
                access_stats["recently_accessed"].append(memory["id"])
            elif days_since_access > 90:
                access_stats["stale_memories"].append(memory["id"])
        
        # Calculate optimization potential
        stale_ratio = len(access_stats["stale_memories"]) / len(memories)
        never_accessed_ratio = len(access_stats["never_accessed"]) / len(memories)
        optimization_potential = (stale_ratio + never_accessed_ratio) / 2
        
        return {
            "access_stats": access_stats,
            "total_accesses": total_accesses,
            "avg_accesses_per_memory": total_accesses / len(memories),
            "stale_ratio": stale_ratio,
            "never_accessed_ratio": never_accessed_ratio,
            "optimization_potential": optimization_potential
        }
    
    def _create_access_optimization_recommendation(
        self,
        access_analysis: Dict[str, Any],
        memories: List[Dict[str, Any]],
        context: OrganizationContext
    ) -> OrganizationRecommendation:
        """Create access pattern optimization recommendation."""
        rec_id = hashlib.md5(f"access_{context.user_id}".encode()).hexdigest()[:16]
        
        optimization_potential = access_analysis["optimization_potential"]
        stale_count = len(access_analysis["access_stats"]["stale_memories"])
        never_accessed_count = len(access_analysis["access_stats"]["never_accessed"])
        
        if optimization_potential > 0.6:
            priority = Priority.HIGH
            impact = ImpactLevel.SIGNIFICANT
            title = "Optimize Memory Access and Archival"
            description = f"{stale_count} stale and {never_accessed_count} never-accessed memories could be archived or reorganized."
        elif optimization_potential > 0.4:
            priority = Priority.MEDIUM
            impact = ImpactLevel.MODERATE
            title = "Improve Memory Access Patterns"
            description = f"Some memories ({stale_count + never_accessed_count}) show low access patterns and could be optimized."
        else:
            priority = Priority.LOW
            impact = ImpactLevel.MINIMAL
            title = "Fine-tune Memory Organization"
            description = "Minor optimizations possible for memory access patterns."
        
        implementation_steps = [
            "Review memories with low access patterns",
            "Archive or remove truly unused memories",
            "Promote frequently accessed memories",
            "Reorganize stale memories into archive",
            "Set up access-based memory suggestions"
        ]
        
        affected_memories = (
            access_analysis["access_stats"]["stale_memories"] +
            access_analysis["access_stats"]["never_accessed"]
        )
        
        current_metrics = {
            "access_efficiency": 1 - optimization_potential,
            "memory_utilization": access_analysis["avg_accesses_per_memory"] / 10,  # Normalize
            "organization_health": 1 - (access_analysis["stale_ratio"] * 0.5)
        }
        
        projected_metrics = {
            "access_efficiency": min(current_metrics["access_efficiency"] + 0.3, 1.0),
            "memory_utilization": min(current_metrics["memory_utilization"] + 0.2, 1.0),
            "organization_health": 0.9
        }
        
        evidence = [
            f"{stale_count} memories not accessed in 90+ days",
            f"{never_accessed_count} memories never accessed",
            f"Optimization potential: {optimization_potential:.2f}"
        ]
        
        benefits = [
            "Reduced clutter from unused memories",
            "Faster access to frequently used content",
            "Better resource utilization",
            "Improved memory maintenance"
        ]
        
        return OrganizationRecommendation(
            recommendation_id=rec_id,
            recommendation_type=RecommendationType.ACCESS_PATTERN_OPTIMIZATION,
            title=title,
            description=description,
            priority=priority,
            impact_level=impact,
            confidence=0.8,
            affected_memories=affected_memories,
            implementation_steps=implementation_steps,
            estimated_effort="1-3 hours",
            expected_benefits=benefits,
            current_metrics=current_metrics,
            projected_metrics=projected_metrics,
            supporting_evidence=evidence,
            tags={"access", "optimization", "archival"}
        )
    
    async def _analyze_duplicates(
        self,
        memories: List[Dict[str, Any]],
        context: OrganizationContext
    ) -> List[OrganizationRecommendation]:
        """Analyze potential duplicate memories."""
        recommendations = []
        
        try:
            if len(memories) < 5:
                return recommendations
            
            # Find potential duplicates using content similarity
            duplicates = await self._find_duplicate_memories(memories)
            
            if duplicates:
                rec = self._create_duplicate_consolidation_recommendation(
                    duplicates, memories, context
                )
                recommendations.append(rec)
                
        except Exception as e:
            logger.error("Error in duplicate analysis", error=str(e))
        
        return recommendations
    
    async def _find_duplicate_memories(self, memories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find potential duplicate memories using similarity analysis."""
        duplicates = []
        
        # Get embeddings for all memories
        embeddings = []
        memory_data = []
        
        for memory in memories:
            embedding = await self._get_or_compute_embedding(memory["content"])
            if embedding is not None:
                embeddings.append(embedding)
                memory_data.append(memory)
        
        if len(embeddings) < 2:
            return duplicates
        
        embeddings_matrix = np.array(embeddings)
        
        # Find similar memory pairs
        for i in range(len(embeddings_matrix)):
            for j in range(i + 1, len(embeddings_matrix)):
                similarity = np.dot(embeddings_matrix[i], embeddings_matrix[j]) / (
                    np.linalg.norm(embeddings_matrix[i]) * np.linalg.norm(embeddings_matrix[j])
                )
                
                if similarity > 0.85:  # High similarity threshold for duplicates
                    # Additional checks for duplicates
                    mem1 = memory_data[i]
                    mem2 = memory_data[j]
                    
                    # Check content length similarity
                    len_ratio = min(len(mem1["content"]), len(mem2["content"])) / max(len(mem1["content"]), len(mem2["content"]))
                    
                    if len_ratio > 0.7:  # Similar lengths
                        duplicates.append({
                            "memory1": mem1,
                            "memory2": mem2,
                            "similarity": similarity,
                            "length_ratio": len_ratio,
                            "confidence": (similarity + len_ratio) / 2
                        })
        
        return duplicates
    
    def _create_duplicate_consolidation_recommendation(
        self,
        duplicates: List[Dict[str, Any]],
        memories: List[Dict[str, Any]],
        context: OrganizationContext
    ) -> OrganizationRecommendation:
        """Create duplicate consolidation recommendation."""
        rec_id = hashlib.md5(f"duplicates_{context.user_id}_{len(duplicates)}".encode()).hexdigest()[:16]
        
        duplicate_count = len(duplicates)
        
        if duplicate_count > 10:
            priority = Priority.HIGH
            impact = ImpactLevel.SIGNIFICANT
            title = "Consolidate Duplicate Memories"
            description = f"Found {duplicate_count} potential duplicate memory pairs that could be consolidated to reduce clutter."
        elif duplicate_count > 5:
            priority = Priority.MEDIUM
            impact = ImpactLevel.MODERATE
            title = "Remove Duplicate Memories"
            description = f"Identified {duplicate_count} potential duplicate pairs that could be merged or removed."
        else:
            priority = Priority.LOW
            impact = ImpactLevel.MINIMAL
            title = "Clean Up Duplicate Content"
            description = f"Found {duplicate_count} potential duplicate(s) for review."
        
        # Get affected memory IDs
        affected_memories = []
        for dup in duplicates:
            affected_memories.extend([dup["memory1"]["id"], dup["memory2"]["id"]])
        affected_memories = list(set(affected_memories))
        
        implementation_steps = [
            "Review identified duplicate memory pairs",
            "Decide which version to keep for each pair",
            "Merge complementary information if needed",
            "Remove or archive duplicate memories",
            "Update tags and references as needed"
        ]
        
        current_metrics = {
            "duplicate_ratio": len(affected_memories) / len(memories),
            "storage_efficiency": 1 - (len(affected_memories) / len(memories) * 0.5),
            "content_quality": 1 - (duplicate_count / len(memories))
        }
        
        projected_metrics = {
            "duplicate_ratio": 0.0,
            "storage_efficiency": 1.0,
            "content_quality": 1.0
        }
        
        evidence = [
            f"{duplicate_count} duplicate pairs identified",
            f"Average similarity: {np.mean([d['similarity'] for d in duplicates]):.3f}",
            f"{len(affected_memories)} memories affected"
        ]
        
        benefits = [
            "Reduced storage and maintenance overhead",
            "Cleaner, more focused memory collection",
            "Eliminated redundant content",
            "Improved search result quality"
        ]
        
        return OrganizationRecommendation(
            recommendation_id=rec_id,
            recommendation_type=RecommendationType.DUPLICATE_CONSOLIDATION,
            title=title,
            description=description,
            priority=priority,
            impact_level=impact,
            confidence=0.9,
            affected_memories=affected_memories,
            implementation_steps=implementation_steps,
            estimated_effort="1-2 hours",
            expected_benefits=benefits,
            current_metrics=current_metrics,
            projected_metrics=projected_metrics,
            supporting_evidence=evidence,
            tags={"duplicates", "consolidation", "cleanup"},
            metadata={"duplicates_found": duplicates}
        )
    
    async def _analyze_metadata_quality(
        self,
        memories: List[Dict[str, Any]],
        context: OrganizationContext
    ) -> List[OrganizationRecommendation]:
        """Analyze metadata quality and suggest enhancements."""
        recommendations = []
        
        try:
            # Analyze metadata completeness and quality
            metadata_analysis = self._analyze_metadata_completeness(memories)
            
            if metadata_analysis["enhancement_potential"] > 0.3:
                rec = self._create_metadata_enhancement_recommendation(
                    metadata_analysis, memories, context
                )
                recommendations.append(rec)
                
        except Exception as e:
            logger.error("Error in metadata analysis", error=str(e))
        
        return recommendations
    
    def _analyze_metadata_completeness(self, memories: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze completeness and quality of memory metadata."""
        metadata_fields = ["tags", "entities", "source", "category", "importance", "context"]
        
        field_completeness = {}
        memories_missing_metadata = []
        
        for field in metadata_fields:
            complete_count = 0
            for memory in memories:
                if field in memory and memory[field]:
                    complete_count += 1
            
            field_completeness[field] = complete_count / len(memories)
        
        # Find memories with poor metadata
        for memory in memories:
            metadata_score = sum(
                1 for field in metadata_fields
                if field in memory and memory[field]
            ) / len(metadata_fields)
            
            if metadata_score < 0.5:
                memories_missing_metadata.append(memory["id"])
        
        avg_completeness = sum(field_completeness.values()) / len(field_completeness)
        enhancement_potential = 1 - avg_completeness
        
        return {
            "field_completeness": field_completeness,
            "avg_completeness": avg_completeness,
            "enhancement_potential": enhancement_potential,
            "memories_missing_metadata": memories_missing_metadata,
            "completeness_distribution": self._calculate_completeness_distribution(memories, metadata_fields)
        }
    
    def _calculate_completeness_distribution(
        self,
        memories: List[Dict[str, Any]],
        metadata_fields: List[str]
    ) -> Dict[str, int]:
        """Calculate distribution of metadata completeness."""
        distribution = {"excellent": 0, "good": 0, "fair": 0, "poor": 0}
        
        for memory in memories:
            completeness = sum(
                1 for field in metadata_fields
                if field in memory and memory[field]
            ) / len(metadata_fields)
            
            if completeness >= 0.8:
                distribution["excellent"] += 1
            elif completeness >= 0.6:
                distribution["good"] += 1
            elif completeness >= 0.4:
                distribution["fair"] += 1
            else:
                distribution["poor"] += 1
        
        return distribution
    
    def _create_metadata_enhancement_recommendation(
        self,
        metadata_analysis: Dict[str, Any],
        memories: List[Dict[str, Any]],
        context: OrganizationContext
    ) -> OrganizationRecommendation:
        """Create metadata enhancement recommendation."""
        rec_id = hashlib.md5(f"metadata_{context.user_id}".encode()).hexdigest()[:16]
        
        enhancement_potential = metadata_analysis["enhancement_potential"]
        missing_count = len(metadata_analysis["memories_missing_metadata"])
        
        if enhancement_potential > 0.6:
            priority = Priority.MEDIUM
            impact = ImpactLevel.SIGNIFICANT
            title = "Enhance Memory Metadata"
            description = f"{missing_count} memories have incomplete metadata. Enhancing metadata would improve searchability and organization."
        else:
            priority = Priority.LOW
            impact = ImpactLevel.MODERATE
            title = "Complete Memory Metadata"
            description = f"Some memories could benefit from richer metadata for better organization."
        
        implementation_steps = [
            "Review memories with incomplete metadata",
            "Add missing tags, categories, and context",
            "Extract and add entity information",
            "Set importance levels for memories",
            "Establish metadata standards for future"
        ]
        
        affected_memories = metadata_analysis["memories_missing_metadata"]
        
        current_metrics = {
            "metadata_completeness": metadata_analysis["avg_completeness"],
            "searchability": metadata_analysis["avg_completeness"] * 0.9,
            "organization_quality": metadata_analysis["avg_completeness"] * 0.8
        }
        
        projected_metrics = {
            "metadata_completeness": min(metadata_analysis["avg_completeness"] + 0.4, 1.0),
            "searchability": min(metadata_analysis["avg_completeness"] + 0.5, 1.0),
            "organization_quality": min(metadata_analysis["avg_completeness"] + 0.3, 1.0)
        }
        
        evidence = [
            f"Average metadata completeness: {metadata_analysis['avg_completeness']:.2f}",
            f"{missing_count} memories need metadata enhancement",
            f"Field completeness: {metadata_analysis['field_completeness']}"
        ]
        
        benefits = [
            "Improved memory searchability",
            "Better categorization and filtering",
            "Enhanced memory discovery",
            "More effective memory management"
        ]
        
        return OrganizationRecommendation(
            recommendation_id=rec_id,
            recommendation_type=RecommendationType.METADATA_ENHANCEMENT,
            title=title,
            description=description,
            priority=priority,
            impact_level=impact,
            confidence=0.8,
            affected_memories=affected_memories,
            implementation_steps=implementation_steps,
            estimated_effort="2-4 hours",
            expected_benefits=benefits,
            current_metrics=current_metrics,
            projected_metrics=projected_metrics,
            supporting_evidence=evidence,
            tags={"metadata", "enhancement", "searchability"}
        )
    
    async def _analyze_relationships(
        self,
        memories: List[Dict[str, Any]],
        context: OrganizationContext
    ) -> List[OrganizationRecommendation]:
        """Analyze memory relationships and suggest strengthening."""
        recommendations = []
        
        try:
            # This would integrate with the connection system
            # For now, provide a simplified analysis
            
            relationship_analysis = await self._analyze_memory_relationships(memories)
            
            if relationship_analysis["improvement_potential"] > 0.4:
                rec = self._create_relationship_strengthening_recommendation(
                    relationship_analysis, memories, context
                )
                recommendations.append(rec)
                
        except Exception as e:
            logger.error("Error in relationship analysis", error=str(e))
        
        return recommendations
    
    async def _analyze_memory_relationships(self, memories: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze relationships between memories."""
        # Simplified relationship analysis
        connected_memories = 0
        total_possible_connections = len(memories) * (len(memories) - 1) // 2
        
        # This would typically use the connection generator
        # For now, estimate based on shared tags and entities
        relationships_found = 0
        
        for i, mem1 in enumerate(memories):
            for mem2 in memories[i+1:]:
                # Check for shared elements
                tags1 = set(mem1.get("tags", []))
                tags2 = set(mem2.get("tags", []))
                entities1 = set(mem1.get("entities", []))
                entities2 = set(mem2.get("entities", []))
                
                if tags1.intersection(tags2) or entities1.intersection(entities2):
                    relationships_found += 1
        
        connection_density = relationships_found / max(total_possible_connections, 1)
        improvement_potential = 1 - connection_density if connection_density < 0.5 else 0
        
        return {
            "relationships_found": relationships_found,
            "connection_density": connection_density,
            "improvement_potential": improvement_potential,
            "total_possible": total_possible_connections
        }
    
    def _create_relationship_strengthening_recommendation(
        self,
        relationship_analysis: Dict[str, Any],
        memories: List[Dict[str, Any]],
        context: OrganizationContext
    ) -> OrganizationRecommendation:
        """Create relationship strengthening recommendation."""
        rec_id = hashlib.md5(f"relationships_{context.user_id}".encode()).hexdigest()[:16]
        
        improvement_potential = relationship_analysis["improvement_potential"]
        connection_density = relationship_analysis["connection_density"]
        
        if improvement_potential > 0.6:
            priority = Priority.MEDIUM
            impact = ImpactLevel.MODERATE
            title = "Strengthen Memory Relationships"
            description = f"Low connection density ({connection_density:.2f}) suggests opportunities to strengthen memory relationships."
        else:
            priority = Priority.LOW
            impact = ImpactLevel.MINIMAL
            title = "Enhance Memory Connections"
            description = "Some memory relationships could be strengthened for better navigation."
        
        implementation_steps = [
            "Review memories for potential connections",
            "Add cross-references between related memories",
            "Create link-based navigation paths",
            "Implement related memory suggestions",
            "Build memory network visualization"
        ]
        
        affected_memories = [memory["id"] for memory in memories]
        
        current_metrics = {
            "connection_density": connection_density,
            "relationship_strength": connection_density * 0.8,
            "network_navigability": connection_density * 0.9
        }
        
        projected_metrics = {
            "connection_density": min(connection_density + 0.3, 1.0),
            "relationship_strength": min(connection_density + 0.4, 1.0),
            "network_navigability": min(connection_density + 0.5, 1.0)
        }
        
        evidence = [
            f"Current connection density: {connection_density:.3f}",
            f"Relationships found: {relationship_analysis['relationships_found']}",
            f"Improvement potential: {improvement_potential:.2f}"
        ]
        
        benefits = [
            "Better memory network navigation",
            "Enhanced discovery of related content",
            "Improved memory associations",
            "Stronger knowledge connections"
        ]
        
        return OrganizationRecommendation(
            recommendation_id=rec_id,
            recommendation_type=RecommendationType.RELATIONSHIP_STRENGTHENING,
            title=title,
            description=description,
            priority=priority,
            impact_level=impact,
            confidence=0.7,
            affected_memories=affected_memories[:20],  # Limit for display
            implementation_steps=implementation_steps,
            estimated_effort="3-6 hours",
            expected_benefits=benefits,
            current_metrics=current_metrics,
            projected_metrics=projected_metrics,
            supporting_evidence=evidence,
            tags={"relationships", "connections", "navigation"}
        )
    
    # Additional helper methods...
    
    def _extract_common_themes(self, texts: List[str]) -> List[str]:
        """Extract common themes from a collection of texts."""
        # Simple keyword extraction
        all_words = []
        for text in texts:
            words = [word.lower().strip('.,!?;:"()[]') for word in text.split() if len(word) > 3]
            all_words.extend(words)
        
        # Filter stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'this', 'that', 'these', 'those', 'is', 'are', 'was', 'were', 'been', 'have', 'has'
        }
        
        filtered_words = [word for word in all_words if word not in stop_words]
        word_counts = Counter(filtered_words)
        
        return [word for word, count in word_counts.most_common(10) if count > 1]
    
    def _calculate_time_span(self, cluster_memories: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate time span for a cluster of memories."""
        if not cluster_memories:
            return {"days": 0, "description": "No memories"}
        
        dates = [mem.get("created_at") for mem in cluster_memories if "created_at" in mem]
        if not dates:
            return {"days": 0, "description": "No dates"}
        
        dates = [d for d in dates if d is not None]
        if len(dates) < 2:
            return {"days": 0, "description": "Single date"}
        
        dates.sort()
        span_days = (dates[-1] - dates[0]).days
        
        if span_days <= 1:
            description = "Same day"
        elif span_days <= 7:
            description = f"{span_days} days"
        elif span_days <= 30:
            description = f"{span_days // 7} weeks"
        elif span_days <= 365:
            description = f"{span_days // 30} months"
        else:
            description = f"{span_days // 365} years"
        
        return {"days": span_days, "description": description}
    
    async def _get_or_compute_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embedding from cache or compute it."""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        
        if text_hash in self.embedding_cache:
            return self.embedding_cache[text_hash]
        
        try:
            embedding = await self.embedder.embed_text(text)
            
            # Cache the embedding
            self.embedding_cache[text_hash] = embedding
            
            # Limit cache size
            if len(self.embedding_cache) > 1000:
                oldest_keys = list(self.embedding_cache.keys())[:100]
                for key in oldest_keys:
                    del self.embedding_cache[key]
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error computing embedding: {e}")
            return None
    
    def _deduplicate_recommendations(
        self,
        recommendations: List[OrganizationRecommendation]
    ) -> List[OrganizationRecommendation]:
        """Remove duplicate recommendations."""
        seen_types = set()
        unique_recommendations = []
        
        for rec in recommendations:
            key = (rec.recommendation_type, rec.title)
            if key not in seen_types:
                seen_types.add(key)
                unique_recommendations.append(rec)
        
        return unique_recommendations
    
    def _prioritize_recommendations(
        self,
        recommendations: List[OrganizationRecommendation],
        context: OrganizationContext
    ) -> List[OrganizationRecommendation]:
        """Prioritize recommendations based on impact and confidence."""
        # Define priority weights
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
        
        # Calculate composite score for each recommendation
        for rec in recommendations:
            priority_score = priority_weights.get(rec.priority, 1)
            impact_score = impact_weights.get(rec.impact_level, 1)
            confidence_score = rec.confidence
            
            # Composite score: priority (40%) + impact (40%) + confidence (20%)
            rec.metadata["composite_score"] = (
                priority_score * 0.4 +
                impact_score * 0.4 +
                confidence_score * 5 * 0.2  # Scale confidence to 1-5
            )
        
        # Sort by composite score
        recommendations.sort(key=lambda x: x.metadata.get("composite_score", 0), reverse=True)
        
        # Filter out low priority if requested
        if not context.include_low_priority:
            recommendations = [
                rec for rec in recommendations 
                if rec.priority not in [Priority.LOW, Priority.OPTIONAL]
            ]
        
        return recommendations
    
    async def _calculate_organization_quality(self, memories: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate overall organization quality metrics."""
        # This would be a comprehensive analysis
        # For now, provide simplified metrics
        
        return {
            "overall_quality": 0.7,  # Placeholder
            "tag_quality": 0.6,
            "structure_quality": 0.8,
            "accessibility": 0.7,
            "maintainability": 0.6
        }
    
    async def _perform_cluster_analysis(self, memories: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform detailed cluster analysis."""
        # Simplified cluster analysis
        return {
            "optimal_clusters": min(len(memories) // 5, 10),
            "current_clustering": "unstructured",
            "cluster_quality": 0.5
        }
    
    async def _analyze_memory_structure(self, memories: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze memory structure."""
        return {
            "hierarchy_depth": 1,  # Flat structure
            "branching_factor": len(memories),
            "structure_type": "flat"
        }
    
    async def _analyze_access_patterns_detailed(
        self,
        memories: List[Dict[str, Any]],
        context: OrganizationContext
    ) -> Dict[str, Any]:
        """Detailed access pattern analysis."""
        return {
            "access_distribution": "uniform",
            "popular_memories": [],
            "access_efficiency": 0.7
        }
    
    def _calculate_improvement_potential(
        self,
        recommendations: List[OrganizationRecommendation]
    ) -> Dict[str, float]:
        """Calculate improvement potential by area."""
        areas = {}
        
        for rec in recommendations:
            impact_weight = {
                ImpactLevel.TRANSFORMATIVE: 1.0,
                ImpactLevel.SIGNIFICANT: 0.8,
                ImpactLevel.MODERATE: 0.6,
                ImpactLevel.MINIMAL: 0.4,
                ImpactLevel.UNCERTAIN: 0.2
            }.get(rec.impact_level, 0.2)
            
            rec_type = rec.recommendation_type.value
            if rec_type not in areas:
                areas[rec_type] = 0.0
            
            areas[rec_type] = max(areas[rec_type], impact_weight * rec.confidence)
        
        return areas
    
    def get_recommendation_analytics(self) -> Dict[str, Any]:
        """Get comprehensive analytics about recommendations."""
        return {
            "stats": self.recommendation_stats.copy(),
            "cache_efficiency": {
                "embedding_cache_size": len(self.embedding_cache),
                "analysis_cache_size": len(self.analysis_cache)
            },
            "performance": {
                "average_processing_time": self.recommendation_stats['average_processing_time'],
                "total_recommendations": self.recommendation_stats['total_recommendations']
            }
        }


# Example usage
async def example_usage():
    """Example of using LocalOrganizationRecommender."""
    from ...core.embeddings.embedder import Embedder
    from ...core.memory.manager import MemoryManager
    
    # Initialize components
    embedder = Embedder()
    memory_manager = MemoryManager()
    
    # Create recommender
    recommender = LocalOrganizationRecommender(embedder, memory_manager)
    
    # Create analysis context
    context = OrganizationContext(
        user_id="user123",
        analysis_scope="all",
        max_recommendations=10,
        include_low_priority=False
    )
    
    # Analyze organization
    analysis = await recommender.analyze_organization(context)
    
    print(f"Generated {len(analysis.recommendations)} recommendations")
    print(f"Processing time: {analysis.processing_time_ms:.2f}ms")
    print(f"Strategies used: {analysis.analysis_strategies}")
    
    for i, rec in enumerate(analysis.recommendations[:5], 1):
        print(f"\n{i}. {rec.title}")
        print(f"   Priority: {rec.priority} | Impact: {rec.impact_level}")
        print(f"   Confidence: {rec.confidence:.2f}")
        print(f"   Description: {rec.description}")
        print(f"   Affected memories: {len(rec.affected_memories)}")
    
    # Get analytics
    analytics = recommender.get_recommendation_analytics()
    print(f"\nAnalytics: {analytics}")


if __name__ == "__main__":
    asyncio.run(example_usage())