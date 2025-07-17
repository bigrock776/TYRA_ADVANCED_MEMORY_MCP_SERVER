"""
Local Knowledge Gap Identification Dashboard.

This module provides comprehensive knowledge gap detection using local algorithms,
impact assessment with local metrics, recommendation generation with heuristics,
and monitoring with local tracking for intelligent knowledge management.
"""

import asyncio
import json
import math
import hashlib
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
import networkx as nx
from collections import defaultdict, Counter
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import spacy
import re

import structlog
from pydantic import BaseModel, Field, ConfigDict, field_validator

from ...core.memory.manager import MemoryManager
from ...core.graph.neo4j_client import Neo4jClient
from ...core.cache.redis_cache import RedisCache
from ...core.utils.config import settings
from ...suggestions.gaps.local_detector import LocalKnowledgeGapDetector, KnowledgeGap, GapType

logger = structlog.get_logger(__name__)


class GapSeverity(str, Enum):
    """Severity levels for knowledge gaps."""
    CRITICAL = "critical"        # Severely impacts understanding
    HIGH = "high"               # Significantly limits effectiveness
    MEDIUM = "medium"           # Moderately important
    LOW = "low"                 # Minor inconvenience
    MINIMAL = "minimal"         # Barely noticeable


class GapCategory(str, Enum):
    """Categories of knowledge gaps."""
    DOMAIN_KNOWLEDGE = "domain_knowledge"        # Missing domain-specific knowledge
    PROCEDURAL_KNOWLEDGE = "procedural_knowledge" # Missing how-to knowledge
    FACTUAL_KNOWLEDGE = "factual_knowledge"      # Missing facts and data
    CONCEPTUAL_KNOWLEDGE = "conceptual_knowledge" # Missing concepts and theories
    CONTEXTUAL_KNOWLEDGE = "contextual_knowledge" # Missing background context
    RELATIONAL_KNOWLEDGE = "relational_knowledge" # Missing connections
    TEMPORAL_KNOWLEDGE = "temporal_knowledge"    # Missing time-based information
    CAUSAL_KNOWLEDGE = "causal_knowledge"        # Missing cause-effect relationships


class RecommendationPriority(str, Enum):
    """Priority levels for gap-filling recommendations."""
    URGENT = "urgent"           # Address immediately
    HIGH = "high"              # Address within days
    MEDIUM = "medium"          # Address within weeks
    LOW = "low"                # Address when convenient
    OPTIONAL = "optional"      # Address if time permits


class LearningPath(BaseModel):
    """Structured learning path to fill knowledge gaps."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    path_id: str = Field(..., description="Unique learning path identifier")
    title: str = Field(..., description="Learning path title")
    description: str = Field(..., description="Detailed description")
    target_gaps: List[str] = Field(..., description="Gap IDs this path addresses")
    
    # Learning steps
    steps: List[Dict[str, Any]] = Field(..., description="Ordered learning steps")
    estimated_duration: str = Field(..., description="Estimated completion time")
    difficulty_level: str = Field(..., description="Difficulty level")
    
    # Prerequisites and outcomes
    prerequisites: List[str] = Field(default_factory=list, description="Required prior knowledge")
    learning_outcomes: List[str] = Field(..., description="Expected outcomes")
    success_metrics: List[str] = Field(..., description="How to measure success")
    
    # Resources
    recommended_resources: List[Dict[str, str]] = Field(..., description="Learning resources")
    practice_exercises: List[str] = Field(default_factory=list, description="Practice activities")
    
    # Metadata
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Creation time")
    created_by: str = Field(..., description="Creator identifier")
    tags: List[str] = Field(default_factory=list, description="Learning path tags")


class GapAnalysisResult(BaseModel):
    """Result of knowledge gap analysis."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    analysis_id: str = Field(..., description="Unique analysis identifier")
    user_id: str = Field(..., description="User being analyzed")
    gaps_identified: List[KnowledgeGap] = Field(..., description="Identified knowledge gaps")
    
    # Analysis summary
    total_gaps: int = Field(..., description="Total number of gaps found")
    critical_gaps: int = Field(..., description="Number of critical gaps")
    coverage_score: float = Field(..., ge=0.0, le=1.0, description="Knowledge coverage score")
    
    # Gap distribution
    gaps_by_category: Dict[str, int] = Field(..., description="Gaps grouped by category")
    gaps_by_severity: Dict[str, int] = Field(..., description="Gaps grouped by severity")
    gaps_by_domain: Dict[str, int] = Field(..., description="Gaps grouped by domain")
    
    # Recommendations
    learning_paths: List[LearningPath] = Field(..., description="Recommended learning paths")
    immediate_actions: List[str] = Field(..., description="Actions to take immediately")
    long_term_goals: List[str] = Field(..., description="Long-term knowledge goals")
    
    # Analysis metadata
    analysis_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="When analysis was performed")
    analysis_parameters: Dict[str, Any] = Field(..., description="Parameters used for analysis")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence in analysis")


@dataclass
class GapAnalysisConfig:
    """Configuration for knowledge gap analysis."""
    user_id: str
    analysis_depth: str = "comprehensive"  # "quick", "standard", "comprehensive"
    focus_domains: Optional[List[str]] = None
    include_minor_gaps: bool = True
    min_confidence_threshold: float = 0.5
    max_gaps_to_analyze: int = 50
    enable_learning_paths: bool = True
    include_resource_recommendations: bool = True
    temporal_scope_days: int = 90


class LocalGapIdentifier:
    """Local Knowledge Gap Identification Dashboard System."""
    
    def __init__(
        self,
        memory_manager: MemoryManager,
        graph_client: Neo4jClient,
        gap_detector: LocalKnowledgeGapDetector,
        cache: Optional[RedisCache] = None
    ):
        self.memory_manager = memory_manager
        self.graph_client = graph_client
        self.gap_detector = gap_detector
        self.cache = cache
        
        # Analysis state
        self.current_analysis: Optional[GapAnalysisResult] = None
        self.analysis_history: List[GapAnalysisResult] = []
        self.learning_paths_cache: Dict[str, List[LearningPath]] = {}
        
        # NLP models
        self.nlp = None
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        
        # Dash app
        self.app = None
        self.is_running = False
        
        logger.info("LocalGapIdentifier initialized")
    
    async def initialize_nlp(self) -> None:
        """Initialize NLP models."""
        try:
            # Load spaCy model
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning("spaCy model not found, using basic text processing")
                self.nlp = None
            
            logger.info("NLP models initialized")
            
        except Exception as e:
            logger.error("Failed to initialize NLP models", error=str(e))
    
    async def analyze_knowledge_gaps(
        self,
        config: GapAnalysisConfig
    ) -> GapAnalysisResult:
        """Perform comprehensive knowledge gap analysis."""
        try:
            # Check cache first
            cache_key = f"gap_analysis:{config.user_id}:{hash(str(config))}"
            if self.cache:
                cached = await self.cache.get(cache_key)
                if cached:
                    logger.info("Loaded gap analysis from cache", user_id=config.user_id)
                    return GapAnalysisResult.model_validate_json(cached)
            
            # Get user memories and knowledge
            memories = await self.memory_manager.get_all_memories(config.user_id)
            relationships = await self.graph_client.get_all_relationships(config.user_id)
            
            # Detect knowledge gaps using the gap detector
            gaps = await self.gap_detector.detect_knowledge_gaps(
                config.user_id,
                focus_domains=config.focus_domains
            )
            
            # Filter gaps by confidence threshold
            filtered_gaps = [
                gap for gap in gaps 
                if gap.confidence >= config.min_confidence_threshold
            ][:config.max_gaps_to_analyze]
            
            # Categorize and analyze gaps
            gap_categories = await self._categorize_gaps(filtered_gaps)
            gap_severities = await self._assess_gap_severities(filtered_gaps, memories)
            domain_distribution = await self._analyze_domain_distribution(filtered_gaps)
            
            # Calculate coverage score
            coverage_score = await self._calculate_coverage_score(memories, filtered_gaps)
            
            # Generate learning paths
            learning_paths = []
            if config.enable_learning_paths:
                learning_paths = await self._generate_learning_paths(
                    filtered_gaps, config
                )
            
            # Generate recommendations
            immediate_actions = await self._generate_immediate_actions(filtered_gaps)
            long_term_goals = await self._generate_long_term_goals(filtered_gaps, domain_distribution)
            
            # Create analysis result
            analysis_result = GapAnalysisResult(
                analysis_id=f"analysis_{config.user_id}_{datetime.now(timezone.utc).isoformat()}",
                user_id=config.user_id,
                gaps_identified=filtered_gaps,
                total_gaps=len(filtered_gaps),
                critical_gaps=len([g for g in filtered_gaps if g.priority.value == "critical"]),
                coverage_score=coverage_score,
                gaps_by_category=gap_categories,
                gaps_by_severity=gap_severities,
                gaps_by_domain=domain_distribution,
                learning_paths=learning_paths,
                immediate_actions=immediate_actions,
                long_term_goals=long_term_goals,
                analysis_parameters=config.__dict__,
                confidence_score=np.mean([g.confidence for g in filtered_gaps]) if filtered_gaps else 0.0
            )
            
            # Cache result
            if self.cache:
                await self.cache.set(
                    cache_key,
                    analysis_result.model_dump_json(),
                    expire_minutes=60
                )
            
            # Store in history
            self.current_analysis = analysis_result
            self.analysis_history.append(analysis_result)
            
            logger.info(
                "Knowledge gap analysis completed",
                user_id=config.user_id,
                gaps_found=len(filtered_gaps),
                coverage_score=coverage_score
            )
            
            return analysis_result
            
        except Exception as e:
            logger.error("Failed to analyze knowledge gaps", error=str(e))
            raise
    
    async def _categorize_gaps(self, gaps: List[KnowledgeGap]) -> Dict[str, int]:
        """Categorize gaps by type."""
        categories = defaultdict(int)
        
        for gap in gaps:
            # Map gap types to categories
            if gap.gap_type in [GapType.CONCEPTUAL_GAP, GapType.FACTUAL_GAP]:
                categories[GapCategory.CONCEPTUAL_KNOWLEDGE.value] += 1
            elif gap.gap_type == GapType.PROCEDURAL_GAP:
                categories[GapCategory.PROCEDURAL_KNOWLEDGE.value] += 1
            elif gap.gap_type == GapType.CONTEXTUAL_GAP:
                categories[GapCategory.CONTEXTUAL_KNOWLEDGE.value] += 1
            elif gap.gap_type == GapType.RELATIONSHIP_GAP:
                categories[GapCategory.RELATIONAL_KNOWLEDGE.value] += 1
            elif gap.gap_type == GapType.TEMPORAL_GAP:
                categories[GapCategory.TEMPORAL_KNOWLEDGE.value] += 1
            elif gap.gap_type == GapType.DOMAIN_GAP:
                categories[GapCategory.DOMAIN_KNOWLEDGE.value] += 1
            else:
                categories[GapCategory.FACTUAL_KNOWLEDGE.value] += 1
        
        return dict(categories)
    
    async def _assess_gap_severities(
        self,
        gaps: List[KnowledgeGap],
        memories: List[Any]
    ) -> Dict[str, int]:
        """Assess severity of knowledge gaps."""
        severities = defaultdict(int)
        
        for gap in gaps:
            # Assess severity based on various factors
            severity_score = 0.0
            
            # Factor 1: Priority level
            priority_weights = {
                "critical": 1.0,
                "high": 0.8,
                "medium": 0.6,
                "low": 0.4,
                "optional": 0.2
            }
            severity_score += priority_weights.get(gap.priority.value, 0.5)
            
            # Factor 2: Impact on other knowledge
            # Gaps that affect many related topics are more severe
            related_count = len([m for m in memories if any(
                topic in m.content.lower() 
                for topic in gap.related_topics[:3]
            )])
            impact_factor = min(1.0, related_count / 10)
            severity_score += impact_factor * 0.5
            
            # Factor 3: Confidence in gap detection
            severity_score += gap.confidence * 0.3
            
            # Convert to severity category
            if severity_score >= 0.9:
                severities[GapSeverity.CRITICAL.value] += 1
            elif severity_score >= 0.7:
                severities[GapSeverity.HIGH.value] += 1
            elif severity_score >= 0.5:
                severities[GapSeverity.MEDIUM.value] += 1
            elif severity_score >= 0.3:
                severities[GapSeverity.LOW.value] += 1
            else:
                severities[GapSeverity.MINIMAL.value] += 1
        
        return dict(severities)
    
    async def _analyze_domain_distribution(self, gaps: List[KnowledgeGap]) -> Dict[str, int]:
        """Analyze distribution of gaps across domains."""
        domains = defaultdict(int)
        
        for gap in gaps:
            # Extract domain from gap title and topics
            gap_domains = gap.related_topics[:2]  # Use first 2 topics as domains
            
            if not gap_domains:
                # Extract from title
                words = gap.title.lower().split()
                domain_keywords = [w for w in words if len(w) > 4 and w.isalpha()]
                gap_domains = domain_keywords[:1]
            
            for domain in gap_domains:
                domains[domain] += 1
        
        # If no clear domains, use general categories
        if not domains:
            domains["general"] = len(gaps)
        
        return dict(domains)
    
    async def _calculate_coverage_score(
        self,
        memories: List[Any],
        gaps: List[KnowledgeGap]
    ) -> float:
        """Calculate knowledge coverage score."""
        try:
            if not memories:
                return 0.0
            
            # Extract topics from memories
            memory_topics = set()
            for memory in memories:
                # Extract key terms from memory content
                words = memory.content.lower().split()
                topics = [w for w in words if len(w) > 4 and w.isalpha()]
                memory_topics.update(topics[:5])  # Top 5 topics per memory
            
            # Extract topics from gaps
            gap_topics = set()
            for gap in gaps:
                gap_topics.update(gap.related_topics[:3])
            
            # Calculate coverage
            if not gap_topics:
                return 1.0  # No gaps means perfect coverage
            
            covered_topics = memory_topics.intersection(gap_topics)
            coverage = len(covered_topics) / len(gap_topics) if gap_topics else 1.0
            
            # Adjust for memory quality and diversity
            memory_diversity = len(memory_topics) / len(memories) if memories else 0
            quality_factor = min(1.0, memory_diversity / 5)  # Normalize to 5 topics per memory
            
            final_score = coverage * 0.7 + quality_factor * 0.3
            return min(1.0, final_score)
            
        except Exception as e:
            logger.error("Failed to calculate coverage score", error=str(e))
            return 0.5  # Default moderate coverage
    
    async def _generate_learning_paths(
        self,
        gaps: List[KnowledgeGap],
        config: GapAnalysisConfig
    ) -> List[LearningPath]:
        """Generate structured learning paths to address knowledge gaps."""
        learning_paths = []
        
        try:
            # Group gaps by domain/topic
            domain_gaps = defaultdict(list)
            for gap in gaps:
                primary_topic = gap.related_topics[0] if gap.related_topics else "general"
                domain_gaps[primary_topic].append(gap)
            
            # Create learning path for each domain
            for domain, domain_gap_list in domain_gaps.items():
                if len(domain_gap_list) >= 2:  # Only create paths for domains with multiple gaps
                    # Sort gaps by priority
                    sorted_gaps = sorted(
                        domain_gap_list,
                        key=lambda g: ["critical", "high", "medium", "low", "optional"].index(g.priority.value)
                    )
                    
                    # Create learning path
                    path = await self._create_domain_learning_path(domain, sorted_gaps, config)
                    learning_paths.append(path)
            
            # Create general learning path for remaining gaps
            remaining_gaps = [
                gap for gap in gaps 
                if not any(gap in path_gaps for path_gaps in domain_gaps.values() if len(path_gaps) >= 2)
            ]
            
            if remaining_gaps:
                general_path = await self._create_general_learning_path(remaining_gaps, config)
                learning_paths.append(general_path)
            
            return learning_paths[:5]  # Limit to 5 paths
            
        except Exception as e:
            logger.error("Failed to generate learning paths", error=str(e))
            return []
    
    async def _create_domain_learning_path(
        self,
        domain: str,
        gaps: List[KnowledgeGap],
        config: GapAnalysisConfig
    ) -> LearningPath:
        """Create learning path for a specific domain."""
        path_id = f"path_{domain}_{config.user_id}_{datetime.now(timezone.utc).strftime('%Y%m%d')}"
        
        # Generate learning steps
        steps = []
        for i, gap in enumerate(gaps):
            step = {
                "step_number": i + 1,
                "title": f"Address {gap.title}",
                "description": gap.description,
                "estimated_time": "1-2 hours",
                "activities": gap.filling_suggestions[:2] if gap.filling_suggestions else [],
                "success_criteria": [f"Understand {topic}" for topic in gap.related_topics[:2]]
            }
            steps.append(step)
        
        # Generate resources
        resources = []
        for gap in gaps:
            for suggestion in gap.filling_suggestions[:1]:  # One resource per gap
                resources.append({
                    "type": suggestion.suggestion_type,
                    "title": suggestion.description[:50] + "...",
                    "description": suggestion.description,
                    "url": "",  # Would be populated with actual resources
                    "difficulty": "intermediate"
                })
        
        return LearningPath(
            path_id=path_id,
            title=f"Master {domain.title()} Knowledge",
            description=f"Comprehensive learning path to address {len(gaps)} knowledge gaps in {domain}",
            target_gaps=[gap.gap_id for gap in gaps],
            steps=steps,
            estimated_duration=f"{len(gaps) * 2}-{len(gaps) * 3} hours",
            difficulty_level="intermediate",
            learning_outcomes=[
                f"Comprehensive understanding of {domain}",
                f"Ability to apply {domain} concepts",
                f"Confidence in {domain} discussions"
            ],
            success_metrics=[
                "Can explain key concepts without reference",
                "Can answer questions about the domain",
                "Can identify connections to other domains"
            ],
            recommended_resources=resources,
            practice_exercises=[
                f"Create a mind map of {domain} concepts",
                f"Write a summary of key {domain} principles",
                f"Find real-world examples of {domain} applications"
            ],
            created_by="gap_analyzer",
            tags=[domain, "knowledge_gap", "learning_path"]
        )
    
    async def _create_general_learning_path(
        self,
        gaps: List[KnowledgeGap],
        config: GapAnalysisConfig
    ) -> LearningPath:
        """Create general learning path for miscellaneous gaps."""
        path_id = f"path_general_{config.user_id}_{datetime.now(timezone.utc).strftime('%Y%m%d')}"
        
        # Prioritize critical gaps
        sorted_gaps = sorted(
            gaps,
            key=lambda g: (
                ["critical", "high", "medium", "low", "optional"].index(g.priority.value),
                -g.confidence
            )
        )
        
        steps = []
        for i, gap in enumerate(sorted_gaps[:5]):  # Limit to 5 most important
            step = {
                "step_number": i + 1,
                "title": f"Address {gap.title}",
                "description": gap.description,
                "estimated_time": "30-60 minutes",
                "activities": gap.filling_suggestions[:1] if gap.filling_suggestions else [],
                "success_criteria": [f"Understand {gap.title}"]
            }
            steps.append(step)
        
        return LearningPath(
            path_id=path_id,
            title="Essential Knowledge Gaps",
            description=f"Address {len(sorted_gaps)} essential knowledge gaps across various domains",
            target_gaps=[gap.gap_id for gap in sorted_gaps],
            steps=steps,
            estimated_duration=f"{len(sorted_gaps)} - {len(sorted_gaps) * 2} hours",
            difficulty_level="mixed",
            learning_outcomes=[
                "Improved overall knowledge coverage",
                "Better understanding of key concepts",
                "Enhanced ability to make connections"
            ],
            success_metrics=[
                "Increased confidence in various topics",
                "Ability to discuss previously unknown concepts",
                "Improved performance in knowledge assessments"
            ],
            recommended_resources=[
                {
                    "type": "online_course",
                    "title": "Foundational Knowledge Course",
                    "description": "Covers essential concepts across multiple domains",
                    "url": "",
                    "difficulty": "beginner"
                }
            ],
            practice_exercises=[
                "Create concept maps for each domain",
                "Write brief explanations of new concepts",
                "Find connections between different knowledge areas"
            ],
            created_by="gap_analyzer",
            tags=["general", "knowledge_gap", "learning_path", "essentials"]
        )
    
    async def _generate_immediate_actions(self, gaps: List[KnowledgeGap]) -> List[str]:
        """Generate immediate actions to address critical gaps."""
        actions = []
        
        # Focus on critical and high priority gaps
        critical_gaps = [g for g in gaps if g.priority.value in ["critical", "high"]]
        
        for gap in critical_gaps[:5]:  # Top 5 critical gaps
            if gap.filling_suggestions:
                suggestion = gap.filling_suggestions[0]
                action = f"📚 {suggestion.description[:80]}..."
                actions.append(action)
            else:
                action = f"🔍 Research {gap.title} to fill critical knowledge gap"
                actions.append(action)
        
        # Add general actions
        if len(gaps) > 5:
            actions.append(f"📊 Review all {len(gaps)} identified knowledge gaps")
        
        if not actions:
            actions.append("✅ No immediate critical actions required")
        
        return actions
    
    async def _generate_long_term_goals(
        self,
        gaps: List[KnowledgeGap],
        domain_distribution: Dict[str, int]
    ) -> List[str]:
        """Generate long-term knowledge goals."""
        goals = []
        
        # Domain-specific goals
        top_domains = sorted(domain_distribution.items(), key=lambda x: x[1], reverse=True)[:3]
        
        for domain, gap_count in top_domains:
            if gap_count >= 2:
                goals.append(f"🎯 Achieve mastery in {domain} (addressing {gap_count} knowledge gaps)")
        
        # Coverage goals
        total_gaps = len(gaps)
        if total_gaps > 10:
            goals.append(f"📈 Reduce knowledge gaps by 50% (from {total_gaps} to {total_gaps//2})")
        
        # Quality goals
        high_confidence_gaps = len([g for g in gaps if g.confidence > 0.8])
        if high_confidence_gaps > 0:
            goals.append(f"🎯 Address all {high_confidence_gaps} high-confidence knowledge gaps")
        
        # General improvement goals
        goals.append("🧠 Improve overall knowledge coverage score by 20%")
        goals.append("🔗 Strengthen connections between different knowledge domains")
        
        return goals[:5]  # Limit to 5 goals
    
    def create_dash_app(self) -> dash.Dash:
        """Create Dash application for gap identification dashboard."""
        app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        
        app.layout = html.Div([
            dbc.Container([
                dbc.Row([
                    dbc.Col([
                        html.H1("Knowledge Gap Identification Dashboard", className="text-center mb-4"),
                        
                        # Analysis controls
                        dbc.Card([
                            dbc.CardBody([
                                dbc.Row([
                                    dbc.Col([
                                        html.Label("User ID:"),
                                        dcc.Input(
                                            id="user-id-input",
                                            type="text",
                                            placeholder="Enter user ID",
                                            value="default_user"
                                        )
                                    ], width=3),
                                    
                                    dbc.Col([
                                        html.Label("Analysis Depth:"),
                                        dcc.Dropdown(
                                            id="depth-dropdown",
                                            options=[
                                                {"label": "Quick", "value": "quick"},
                                                {"label": "Standard", "value": "standard"},
                                                {"label": "Comprehensive", "value": "comprehensive"}
                                            ],
                                            value="standard"
                                        )
                                    ], width=3),
                                    
                                    dbc.Col([
                                        html.Label("Focus Domains:"),
                                        dcc.Input(
                                            id="domains-input",
                                            type="text",
                                            placeholder="e.g., technology, science",
                                            value=""
                                        )
                                    ], width=4),
                                    
                                    dbc.Col([
                                        dbc.Button(
                                            "Analyze Gaps",
                                            id="analyze-btn",
                                            color="primary",
                                            className="mt-4"
                                        )
                                    ], width=2)
                                ]),
                                
                                dbc.Row([
                                    dbc.Col([
                                        dbc.Switch(
                                            id="include-minor-switch",
                                            label="Include Minor Gaps",
                                            value=True
                                        )
                                    ], width=3),
                                    
                                    dbc.Col([
                                        dbc.Switch(
                                            id="learning-paths-switch",
                                            label="Generate Learning Paths",
                                            value=True
                                        )
                                    ], width=3),
                                    
                                    dbc.Col([
                                        html.Label("Confidence Threshold:"),
                                        dcc.Slider(
                                            id="confidence-slider",
                                            min=0.0,
                                            max=1.0,
                                            step=0.1,
                                            value=0.5,
                                            marks={i/10: f"{i/10:.1f}" for i in range(0, 11, 2)}
                                        )
                                    ], width=6)
                                ], className="mt-3")
                            ])
                        ], className="mb-4"),
                        
                        # Results tabs
                        dbc.Tabs([
                            dbc.Tab(label="Gap Overview", tab_id="overview-tab"),
                            dbc.Tab(label="Gap Details", tab_id="details-tab"),
                            dbc.Tab(label="Learning Paths", tab_id="paths-tab"),
                            dbc.Tab(label="Recommendations", tab_id="recommendations-tab"),
                            dbc.Tab(label="Progress Tracking", tab_id="progress-tab")
                        ], id="results-tabs", active_tab="overview-tab"),
                        
                        # Tab content
                        html.Div(id="tab-content", className="mt-4"),
                        
                        # Analysis status
                        dbc.Alert(
                            id="status-alert",
                            children="Ready to analyze knowledge gaps",
                            color="info",
                            className="mt-4"
                        )
                        
                    ], width=12)
                ])
            ], fluid=True),
            
            # Data stores
            dcc.Store(id="analysis-store"),
            dcc.Store(id="config-store"),
            
            # Auto-refresh
            dcc.Interval(
                id="refresh-interval",
                interval=300000,  # 5 minutes
                n_intervals=0,
                disabled=True
            )
        ])
        
        # Setup callbacks
        self._setup_gap_callbacks(app)
        
        return app
    
    def _setup_gap_callbacks(self, app: dash.Dash) -> None:
        """Setup Dash application callbacks."""
        
        @app.callback(
            [Output("analysis-store", "data"),
             Output("config-store", "data"),
             Output("status-alert", "children"),
             Output("status-alert", "color")],
            [Input("analyze-btn", "n_clicks")],
            [State("user-id-input", "value"),
             State("depth-dropdown", "value"),
             State("domains-input", "value"),
             State("include-minor-switch", "value"),
             State("learning-paths-switch", "value"),
             State("confidence-slider", "value")]
        )
        def perform_gap_analysis(n_clicks, user_id, depth, domains, include_minor, learning_paths, confidence):
            if not n_clicks or not user_id:
                raise PreventUpdate
            
            try:
                # Parse domains
                focus_domains = [d.strip() for d in domains.split(",")] if domains else None
                
                # Create configuration
                config = GapAnalysisConfig(
                    user_id=user_id,
                    analysis_depth=depth,
                    focus_domains=focus_domains,
                    include_minor_gaps=include_minor,
                    min_confidence_threshold=confidence,
                    enable_learning_paths=learning_paths
                )
                
                # For demo purposes, create sample analysis
                analysis_result = self._create_sample_analysis(config)
                
                return (
                    analysis_result.model_dump(),
                    config.__dict__,
                    f"Analysis completed: {analysis_result.total_gaps} gaps found",
                    "success"
                )
                
            except Exception as e:
                return {}, {}, f"Analysis failed: {str(e)}", "danger"
        
        @app.callback(
            Output("tab-content", "children"),
            [Input("results-tabs", "active_tab"),
             Input("analysis-store", "data")]
        )
        def update_tab_content(active_tab, analysis_data):
            if not analysis_data:
                return html.Div("No analysis data available. Run an analysis first.")
            
            try:
                analysis = GapAnalysisResult.model_validate(analysis_data)
                
                if active_tab == "overview-tab":
                    return self._create_overview_tab(analysis)
                elif active_tab == "details-tab":
                    return self._create_details_tab(analysis)
                elif active_tab == "paths-tab":
                    return self._create_paths_tab(analysis)
                elif active_tab == "recommendations-tab":
                    return self._create_recommendations_tab(analysis)
                elif active_tab == "progress-tab":
                    return self._create_progress_tab(analysis)
                else:
                    return html.Div("Invalid tab selected")
                    
            except Exception as e:
                return html.Div(f"Error loading tab content: {str(e)}")
    
    def _create_sample_analysis(self, config: GapAnalysisConfig) -> GapAnalysisResult:
        """Create sample analysis for demo purposes."""
        # This would be replaced with actual analysis in production
        sample_gaps = [
            {
                "gap_id": "gap_1",
                "title": "Machine Learning Fundamentals",
                "description": "Missing foundational knowledge in machine learning concepts",
                "gap_type": "conceptual_gap",
                "priority": "high",
                "confidence": 0.85,
                "related_topics": ["machine learning", "algorithms", "data science"],
                "filling_suggestions": []
            },
            {
                "gap_id": "gap_2", 
                "title": "Database Optimization",
                "description": "Limited understanding of database performance tuning",
                "gap_type": "procedural_gap",
                "priority": "medium",
                "confidence": 0.7,
                "related_topics": ["databases", "performance", "optimization"],
                "filling_suggestions": []
            }
        ]
        
        return GapAnalysisResult(
            analysis_id=f"sample_{config.user_id}",
            user_id=config.user_id,
            gaps_identified=[],  # Would contain actual KnowledgeGap objects
            total_gaps=len(sample_gaps),
            critical_gaps=1,
            coverage_score=0.75,
            gaps_by_category={"conceptual_knowledge": 1, "procedural_knowledge": 1},
            gaps_by_severity={"high": 1, "medium": 1},
            gaps_by_domain={"technology": 2},
            learning_paths=[],
            immediate_actions=[
                "📚 Study machine learning fundamentals",
                "🔍 Research database optimization techniques"
            ],
            long_term_goals=[
                "🎯 Achieve proficiency in data science",
                "📈 Improve technical knowledge coverage by 30%"
            ],
            analysis_parameters=config.__dict__,
            confidence_score=0.775
        )
    
    def _create_overview_tab(self, analysis: GapAnalysisResult) -> html.Div:
        """Create overview tab content."""
        return html.Div([
            dbc.Row([
                # Summary stats
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Analysis Summary", className="card-title"),
                            html.P(f"Total Gaps: {analysis.total_gaps}"),
                            html.P(f"Critical Gaps: {analysis.critical_gaps}"),
                            html.P(f"Coverage Score: {analysis.coverage_score:.2f}"),
                            html.P(f"Confidence: {analysis.confidence_score:.2f}")
                        ])
                    ])
                ], width=6),
                
                # Coverage visualization
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Knowledge Coverage", className="card-title"),
                            dcc.Graph(
                                figure=self._create_coverage_gauge(analysis.coverage_score)
                            )
                        ])
                    ])
                ], width=6)
            ], className="mb-4"),
            
            dbc.Row([
                # Gap distribution by category
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Gaps by Category", className="card-title"),
                            dcc.Graph(
                                figure=self._create_category_chart(analysis.gaps_by_category)
                            )
                        ])
                    ])
                ], width=6),
                
                # Gap distribution by severity
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Gaps by Severity", className="card-title"),
                            dcc.Graph(
                                figure=self._create_severity_chart(analysis.gaps_by_severity)
                            )
                        ])
                    ])
                ], width=6)
            ])
        ])
    
    def _create_details_tab(self, analysis: GapAnalysisResult) -> html.Div:
        """Create gap details tab content."""
        return html.Div([
            html.H4("Detailed Gap Analysis"),
            html.P(f"Found {analysis.total_gaps} knowledge gaps requiring attention."),
            
            # Would show actual gap details here
            dbc.Card([
                dbc.CardBody([
                    html.H5("Sample Gap: Machine Learning Fundamentals"),
                    html.P("Missing foundational knowledge in machine learning concepts and algorithms."),
                    dbc.Badge("High Priority", color="warning"),
                    dbc.Badge("85% Confidence", color="info", className="ms-2"),
                    html.Hr(),
                    html.H6("Related Topics:"),
                    html.Ul([
                        html.Li("Machine Learning"),
                        html.Li("Algorithms"),
                        html.Li("Data Science")
                    ])
                ])
            ], className="mb-3")
        ])
    
    def _create_paths_tab(self, analysis: GapAnalysisResult) -> html.Div:
        """Create learning paths tab content."""
        return html.Div([
            html.H4("Recommended Learning Paths"),
            html.P(f"Generated {len(analysis.learning_paths)} learning paths to address identified gaps."),
            
            # Would show actual learning paths here
            dbc.Card([
                dbc.CardBody([
                    html.H5("🎯 Essential Knowledge Path"),
                    html.P("A structured path to address critical knowledge gaps."),
                    html.P("📅 Estimated Duration: 4-6 hours"),
                    html.P("📊 Difficulty: Intermediate"),
                    html.H6("Learning Steps:"),
                    html.Ol([
                        html.Li("Study machine learning fundamentals"),
                        html.Li("Practice with real datasets"),
                        html.Li("Apply concepts to projects")
                    ])
                ])
            ])
        ])
    
    def _create_recommendations_tab(self, analysis: GapAnalysisResult) -> html.Div:
        """Create recommendations tab content."""
        return html.Div([
            dbc.Row([
                dbc.Col([
                    html.H4("Immediate Actions"),
                    html.Div([
                        dbc.Alert(action, color="warning", className="mb-2")
                        for action in analysis.immediate_actions
                    ])
                ], width=6),
                
                dbc.Col([
                    html.H4("Long-term Goals"),
                    html.Div([
                        dbc.Alert(goal, color="info", className="mb-2")
                        for goal in analysis.long_term_goals
                    ])
                ], width=6)
            ])
        ])
    
    def _create_progress_tab(self, analysis: GapAnalysisResult) -> html.Div:
        """Create progress tracking tab content."""
        return html.Div([
            html.H4("Progress Tracking"),
            html.P("Track your progress in addressing knowledge gaps over time."),
            
            # Progress charts would go here
            dbc.Card([
                dbc.CardBody([
                    html.H5("Gap Resolution Progress"),
                    html.P("This feature tracks how gaps are being addressed over time."),
                    dcc.Graph(
                        figure=self._create_progress_chart()
                    )
                ])
            ])
        ])
    
    def _create_coverage_gauge(self, coverage_score: float) -> go.Figure:
        """Create coverage gauge chart."""
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=coverage_score * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Knowledge Coverage %"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 80], 'color': "yellow"},
                    {'range': [80, 100], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        fig.update_layout(height=300)
        return fig
    
    def _create_category_chart(self, gaps_by_category: Dict[str, int]) -> go.Figure:
        """Create category distribution chart."""
        if not gaps_by_category:
            return go.Figure()
        
        fig = go.Figure(data=[
            go.Pie(
                labels=list(gaps_by_category.keys()),
                values=list(gaps_by_category.values()),
                hole=0.3
            )
        ])
        
        fig.update_layout(height=300)
        return fig
    
    def _create_severity_chart(self, gaps_by_severity: Dict[str, int]) -> go.Figure:
        """Create severity distribution chart."""
        if not gaps_by_severity:
            return go.Figure()
        
        colors = {
            'critical': '#dc3545',
            'high': '#fd7e14', 
            'medium': '#ffc107',
            'low': '#20c997',
            'minimal': '#6c757d'
        }
        
        fig = go.Figure(data=[
            go.Bar(
                x=list(gaps_by_severity.keys()),
                y=list(gaps_by_severity.values()),
                marker_color=[colors.get(k, '#6c757d') for k in gaps_by_severity.keys()]
            )
        ])
        
        fig.update_layout(height=300, xaxis_title="Severity", yaxis_title="Number of Gaps")
        return fig
    
    def _create_progress_chart(self) -> go.Figure:
        """Create progress tracking chart."""
        # Sample progress data
        dates = pd.date_range(start='2024-01-01', periods=10, freq='W')
        gaps_remaining = [20, 18, 17, 15, 13, 12, 10, 8, 6, 5]
        
        fig = go.Figure(data=[
            go.Scatter(
                x=dates,
                y=gaps_remaining,
                mode='lines+markers',
                name='Gaps Remaining',
                line=dict(color='#dc3545', width=3)
            )
        ])
        
        fig.update_layout(
            title="Knowledge Gap Resolution Over Time",
            xaxis_title="Date",
            yaxis_title="Gaps Remaining",
            height=300
        )
        
        return fig
    
    async def start_gap_server(self, host: str = "127.0.0.1", port: int = 8052) -> None:
        """Start the gap identification dashboard server."""
        if self.is_running:
            logger.warning("Gap identification server already running")
            return
        
        try:
            await self.initialize_nlp()
            self.app = self.create_dash_app()
            self.is_running = True
            
            logger.info(
                "Starting gap identification server",
                host=host,
                port=port
            )
            
            # Run in separate thread
            import threading
            server_thread = threading.Thread(
                target=lambda: self.app.run_server(
                    host=host,
                    port=port,
                    debug=False,
                    use_reloader=False
                )
            )
            server_thread.daemon = True
            server_thread.start()
            
        except Exception as e:
            logger.error("Failed to start gap identification server", error=str(e))
            self.is_running = False
            raise


# Module exports
__all__ = [
    "LocalGapIdentifier",
    "GapAnalysisResult",
    "LearningPath",
    "GapAnalysisConfig",
    "GapSeverity",
    "GapCategory",
    "RecommendationPriority"
]