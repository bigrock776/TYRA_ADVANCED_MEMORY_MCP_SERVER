"""
Memory Confidence Agent for Tyra Web Memory System.

Advanced confidence scoring system using Pydantic AI for structured validation,
multi-factor analysis including embedding similarity, freshness assessment,
source domain rating, and comprehensive reasoning generation.
"""

import asyncio
import hashlib
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
from enum import Enum
from urllib.parse import urlparse
import math

import structlog
from pydantic import BaseModel, Field, ConfigDict, field_validator
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel

from ..core.utils.config import settings
from ..memory.neo4j_linker import Neo4jLinker

logger = structlog.get_logger(__name__)


class ConfidenceLevel(str, Enum):
    """Confidence levels for memory validation."""
    ROCK_SOLID = "rock_solid"  # 95-100% - Safe for automated actions
    HIGH = "high"              # 80-94% - Generally reliable
    FUZZY = "fuzzy"            # 60-79% - Needs verification
    LOW = "low"                # 0-59% - Not confident


class FactorType(str, Enum):
    """Types of factors considered in confidence scoring."""
    EMBEDDING_SIMILARITY = "embedding_similarity"
    FRESHNESS = "freshness"
    SOURCE_DOMAIN = "source_domain"
    CONTENT_QUALITY = "content_quality"
    EXTRACTION_METHOD = "extraction_method"
    CROSS_VALIDATION = "cross_validation"
    TEMPORAL_CONSISTENCY = "temporal_consistency"
    AUTHORITY_SIGNALS = "authority_signals"


class ConfidenceFactor(BaseModel):
    """Individual factor contributing to confidence score."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    factor_type: FactorType = Field(..., description="Type of confidence factor")
    score: float = Field(..., ge=0.0, le=1.0, description="Factor score (0-1)")
    weight: float = Field(..., ge=0.0, le=1.0, description="Factor weight in final calculation")
    reasoning: str = Field(..., description="Human-readable reasoning for this factor")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional factor metadata")
    
    @field_validator('score', 'weight')
    @classmethod
    def validate_scores(cls, v):
        """Ensure scores are properly bounded."""
        return max(0.0, min(1.0, v))


class ConfidenceResult(BaseModel):
    """Complete confidence assessment result."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    overall_score: float = Field(..., ge=0.0, le=1.0, description="Overall confidence score (0-1)")
    confidence_level: ConfidenceLevel = Field(..., description="Categorical confidence level")
    factors: List[ConfidenceFactor] = Field(..., description="Individual confidence factors")
    reasoning: str = Field(..., description="Comprehensive reasoning summary")
    recommendations: List[str] = Field(default_factory=list, description="Action recommendations")
    warnings: List[str] = Field(default_factory=list, description="Potential issues or warnings")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Assessment timestamp")
    
    @field_validator('overall_score')
    @classmethod
    def validate_overall_score(cls, v):
        """Ensure overall score is properly bounded."""
        return max(0.0, min(1.0, v))
    
    def is_trading_safe(self) -> bool:
        """Check if confidence is high enough for automated trading decisions."""
        return self.confidence_level == ConfidenceLevel.ROCK_SOLID and self.overall_score >= 0.95


class ContentMetrics(BaseModel):
    """Metrics extracted from content for confidence assessment."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    text_length: int = Field(..., ge=0, description="Content text length")
    sentence_count: int = Field(..., ge=0, description="Number of sentences")
    word_count: int = Field(..., ge=0, description="Number of words")
    paragraph_count: int = Field(..., ge=0, description="Number of paragraphs")
    readability_score: float = Field(..., ge=0.0, le=100.0, description="Readability score")
    technical_terms_count: int = Field(..., ge=0, description="Count of technical terms")
    external_links_count: int = Field(..., ge=0, description="Count of external links")
    citation_count: int = Field(..., ge=0, description="Count of citations or references")


class DomainReputation(BaseModel):
    """Domain reputation metrics."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    domain: str = Field(..., description="Domain name")
    trust_score: float = Field(..., ge=0.0, le=1.0, description="Domain trust score")
    authority_score: float = Field(..., ge=0.0, le=1.0, description="Domain authority score")
    reputation_category: str = Field(..., description="Domain reputation category")
    last_updated: datetime = Field(..., description="Last reputation update")
    source_count: int = Field(..., ge=0, description="Number of sources from this domain")
    
    @field_validator('trust_score', 'authority_score')
    @classmethod
    def validate_scores(cls, v):
        """Ensure scores are properly bounded."""
        return max(0.0, min(1.0, v))


class MemoryConfidenceAgent:
    """
    Advanced Memory Confidence Agent using Pydantic AI.
    
    Provides comprehensive confidence scoring for memory content using multiple
    factors including embedding similarity, freshness, source reputation,
    content quality, and cross-validation with structured reasoning.
    """
    
    def __init__(
        self,
        neo4j_linker: Optional[Neo4jLinker] = None,
        enable_ai_reasoning: bool = True,
        default_weights: Optional[Dict[FactorType, float]] = None
    ):
        """
        Initialize Memory Confidence Agent.
        
        Args:
            neo4j_linker: Neo4j linker for domain reputation lookup
            enable_ai_reasoning: Whether to use AI for reasoning generation
            default_weights: Default weights for confidence factors
        """
        self.neo4j_linker = neo4j_linker
        self.enable_ai_reasoning = enable_ai_reasoning
        
        # Default factor weights
        self.factor_weights = default_weights or {
            FactorType.EMBEDDING_SIMILARITY: 0.25,
            FactorType.FRESHNESS: 0.20,
            FactorType.SOURCE_DOMAIN: 0.20,
            FactorType.CONTENT_QUALITY: 0.15,
            FactorType.EXTRACTION_METHOD: 0.10,
            FactorType.CROSS_VALIDATION: 0.05,
            FactorType.TEMPORAL_CONSISTENCY: 0.03,
            FactorType.AUTHORITY_SIGNALS: 0.02
        }
        
        # Domain reputation cache
        self.domain_cache: Dict[str, DomainReputation] = {}
        
        # Known high-authority domains
        self.authority_domains = {
            'wikipedia.org': 0.95,
            'github.com': 0.90,
            'stackoverflow.com': 0.85,
            'arxiv.org': 0.95,
            'pubmed.ncbi.nlm.nih.gov': 0.98,
            'nature.com': 0.95,
            'sciencedirect.com': 0.90,
            'ieee.org': 0.92,
            'acm.org': 0.88,
            'microsoft.com': 0.85,
            'google.com': 0.80,
            'openai.com': 0.80,
            'anthropic.com': 0.80,
        }
        
        # Initialize Pydantic AI agent if enabled
        self.ai_agent = None
        if enable_ai_reasoning:
            self._initialize_ai_agent()
        
        # Performance tracking
        self.assessment_stats = {
            'total_assessments': 0,
            'high_confidence_count': 0,
            'low_confidence_count': 0,
            'average_assessment_time': 0.0,
            'ai_reasoning_enabled': enable_ai_reasoning
        }
        
        logger.info(
            "Initialized MemoryConfidenceAgent",
            ai_reasoning_enabled=enable_ai_reasoning,
            factor_weights=self.factor_weights
        )
    
    def _initialize_ai_agent(self) -> None:
        """Initialize Pydantic AI agent for reasoning generation."""
        try:
            # Use local LLM through OpenAI-compatible API
            # This assumes a local LLM server is running on localhost:8000
            model = OpenAIModel(
                'llama2',  # Model name
                base_url='http://localhost:8000/v1',  # Local LLM server
                api_key='not-needed'  # Local server doesn't need API key
            )
            
            self.ai_agent = Agent(
                model,
                system_prompt="""
                You are an expert AI system for analyzing the confidence and reliability of information.
                Your task is to provide clear, structured reasoning about why certain information
                should or should not be trusted based on multiple factors.
                
                Focus on:
                1. Source credibility and domain authority
                2. Content quality indicators
                3. Temporal relevance and freshness
                4. Cross-validation opportunities
                5. Potential risks or concerns
                
                Always provide actionable recommendations and clear warnings when appropriate.
                Be concise but thorough in your analysis.
                """,
                retries=2
            )
            
            logger.info("Pydantic AI agent initialized successfully")
            
        except Exception as e:
            logger.warning("Failed to initialize AI agent, using fallback reasoning", error=str(e))
            self.ai_agent = None
    
    async def calculate_confidence(
        self,
        text: str,
        source: str,
        metadata: Optional[Dict[str, Any]] = None,
        query_embedding: Optional[List[float]] = None,
        content_embedding: Optional[List[float]] = None
    ) -> float:
        """
        Calculate confidence score for memory content.
        
        Args:
            text: Content text
            source: Source URL or identifier
            metadata: Additional metadata
            query_embedding: Query embedding for similarity calculation
            content_embedding: Content embedding for similarity calculation
            
        Returns:
            Confidence score (0.0 to 1.0)
        """
        result = await self.assess_confidence(
            text=text,
            source=source,
            metadata=metadata,
            query_embedding=query_embedding,
            content_embedding=content_embedding
        )
        return result.overall_score
    
    async def assess_confidence(
        self,
        text: str,
        source: str,
        metadata: Optional[Dict[str, Any]] = None,
        query_embedding: Optional[List[float]] = None,
        content_embedding: Optional[List[float]] = None
    ) -> ConfidenceResult:
        """
        Perform comprehensive confidence assessment.
        
        Args:
            text: Content text
            source: Source URL or identifier
            metadata: Additional metadata
            query_embedding: Query embedding for similarity calculation
            content_embedding: Content embedding for similarity calculation
            
        Returns:
            Complete confidence assessment result
        """
        start_time = datetime.utcnow()
        metadata = metadata or {}
        
        try:
            # Extract content metrics
            content_metrics = self._extract_content_metrics(text)
            
            # Get domain reputation
            domain_reputation = await self._get_domain_reputation(source)
            
            # Calculate individual confidence factors
            factors = []
            
            # 1. Embedding similarity factor
            if query_embedding and content_embedding:
                similarity_factor = await self._calculate_similarity_factor(
                    query_embedding, content_embedding
                )
                factors.append(similarity_factor)
            
            # 2. Freshness factor
            freshness_factor = self._calculate_freshness_factor(metadata)
            factors.append(freshness_factor)
            
            # 3. Source domain factor
            domain_factor = self._calculate_domain_factor(domain_reputation)
            factors.append(domain_factor)
            
            # 4. Content quality factor
            quality_factor = self._calculate_content_quality_factor(content_metrics, text)
            factors.append(quality_factor)
            
            # 5. Extraction method factor
            extraction_factor = self._calculate_extraction_method_factor(metadata)
            factors.append(extraction_factor)
            
            # 6. Cross-validation factor
            cross_validation_factor = await self._calculate_cross_validation_factor(text, source)
            factors.append(cross_validation_factor)
            
            # 7. Temporal consistency factor
            temporal_factor = self._calculate_temporal_consistency_factor(metadata)
            factors.append(temporal_factor)
            
            # 8. Authority signals factor
            authority_factor = self._calculate_authority_signals_factor(text, source)
            factors.append(authority_factor)
            
            # Calculate overall confidence score
            overall_score = self._calculate_weighted_score(factors)
            
            # Determine confidence level
            confidence_level = self._determine_confidence_level(overall_score)
            
            # Generate reasoning and recommendations
            reasoning, recommendations, warnings = await self._generate_reasoning(
                factors, overall_score, confidence_level, text, source, metadata
            )
            
            # Create result
            result = ConfidenceResult(
                overall_score=overall_score,
                confidence_level=confidence_level,
                factors=factors,
                reasoning=reasoning,
                recommendations=recommendations,
                warnings=warnings
            )
            
            # Update statistics
            assessment_time = (datetime.utcnow() - start_time).total_seconds()
            self.assessment_stats['total_assessments'] += 1
            self.assessment_stats['average_assessment_time'] = (
                (self.assessment_stats['average_assessment_time'] * 
                 (self.assessment_stats['total_assessments'] - 1) + assessment_time) / 
                self.assessment_stats['total_assessments']
            )
            
            if confidence_level in [ConfidenceLevel.HIGH, ConfidenceLevel.ROCK_SOLID]:
                self.assessment_stats['high_confidence_count'] += 1
            elif confidence_level == ConfidenceLevel.LOW:
                self.assessment_stats['low_confidence_count'] += 1
            
            logger.info(
                "Confidence assessment completed",
                source=source[:100],
                overall_score=overall_score,
                confidence_level=confidence_level.value,
                factors_count=len(factors),
                assessment_time_seconds=assessment_time
            )
            
            return result
            
        except Exception as e:
            logger.error("Confidence assessment failed", error=str(e))
            
            # Return low confidence result
            return ConfidenceResult(
                overall_score=0.1,
                confidence_level=ConfidenceLevel.LOW,
                factors=[],
                reasoning=f"Assessment failed due to error: {str(e)}",
                recommendations=["Manual review required"],
                warnings=["Automatic confidence assessment failed"]
            )
    
    def _extract_content_metrics(self, text: str) -> ContentMetrics:
        """Extract metrics from content text."""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        words = text.split()
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        # Simple readability score (Flesch-like)
        avg_sentence_length = len(words) / max(len(sentences), 1)
        avg_syllables = sum(self._count_syllables(word) for word in words) / max(len(words), 1)
        readability = max(0, min(100, 206.835 - 1.015 * avg_sentence_length - 84.6 * avg_syllables))
        
        # Count technical terms (words with 3+ syllables)
        technical_terms = sum(1 for word in words if self._count_syllables(word) >= 3)
        
        # Count external links
        external_links = len(re.findall(r'https?://[^\s]+', text))
        
        # Count citations (rough heuristic)
        citations = len(re.findall(r'\[[0-9]+\]|\([0-9]{4}\)|et al\.', text))
        
        return ContentMetrics(
            text_length=len(text),
            sentence_count=len(sentences),
            word_count=len(words),
            paragraph_count=len(paragraphs),
            readability_score=readability,
            technical_terms_count=technical_terms,
            external_links_count=external_links,
            citation_count=citations
        )
    
    def _count_syllables(self, word: str) -> int:
        """Simple syllable counting heuristic."""
        word = word.lower().strip('.,!?;:"')
        if len(word) <= 3:
            return 1
        
        vowels = 'aeiouy'
        syllables = 0
        prev_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_was_vowel:
                syllables += 1
            prev_was_vowel = is_vowel
        
        # Handle silent e
        if word.endswith('e'):
            syllables -= 1
        
        return max(1, syllables)
    
    async def _get_domain_reputation(self, source: str) -> DomainReputation:
        """Get or calculate domain reputation."""
        try:
            parsed_url = urlparse(source)
            domain = parsed_url.netloc.lower()
            
            # Check cache
            if domain in self.domain_cache:
                cached = self.domain_cache[domain]
                # Use cached if less than 24 hours old
                if datetime.utcnow() - cached.last_updated < timedelta(hours=24):
                    return cached
            
            # Get trust score from Neo4j if available
            neo4j_trust = 0.5  # Default
            if self.neo4j_linker:
                try:
                    neo4j_trust = await self.neo4j_linker.get_domain_trust_score(domain)
                except:
                    pass
            
            # Get authority score from known domains or heuristics
            authority_score = self.authority_domains.get(domain, self._calculate_domain_authority(domain))
            
            # Determine reputation category
            combined_score = (neo4j_trust + authority_score) / 2
            if combined_score >= 0.9:
                category = "highly_trusted"
            elif combined_score >= 0.7:
                category = "trusted"
            elif combined_score >= 0.5:
                category = "neutral"
            elif combined_score >= 0.3:
                category = "questionable"
            else:
                category = "untrusted"
            
            reputation = DomainReputation(
                domain=domain,
                trust_score=neo4j_trust,
                authority_score=authority_score,
                reputation_category=category,
                last_updated=datetime.utcnow(),
                source_count=1  # Would be updated from actual data
            )
            
            # Cache result
            self.domain_cache[domain] = reputation
            
            return reputation
            
        except Exception as e:
            logger.warning("Failed to get domain reputation", source=source, error=str(e))
            
            # Return default neutral reputation
            return DomainReputation(
                domain="unknown",
                trust_score=0.5,
                authority_score=0.5,
                reputation_category="neutral",
                last_updated=datetime.utcnow(),
                source_count=0
            )
    
    def _calculate_domain_authority(self, domain: str) -> float:
        """Calculate domain authority using heuristics."""
        authority = 0.5  # Base authority
        
        # Government domains
        if domain.endswith('.gov'):
            authority = 0.9
        # Education domains
        elif domain.endswith('.edu'):
            authority = 0.85
        # Organization domains
        elif domain.endswith('.org'):
            authority = 0.7
        # Commercial domains
        elif domain.endswith('.com'):
            authority = 0.6
        
        # Well-known subdomains
        if any(subdomain in domain for subdomain in ['docs.', 'help.', 'support.', 'api.']):
            authority += 0.1
        
        # Suspicious indicators
        if any(indicator in domain for indicator in ['free', 'blog', 'personal', 'temp']):
            authority -= 0.2
        
        return max(0.0, min(1.0, authority))
    
    async def _calculate_similarity_factor(
        self,
        query_embedding: List[float],
        content_embedding: List[float]
    ) -> ConfidenceFactor:
        """Calculate embedding similarity confidence factor."""
        try:
            # Calculate cosine similarity
            dot_product = sum(a * b for a, b in zip(query_embedding, content_embedding))
            norm_a = math.sqrt(sum(a * a for a in query_embedding))
            norm_b = math.sqrt(sum(b * b for b in content_embedding))
            
            similarity = dot_product / (norm_a * norm_b) if norm_a * norm_b > 0 else 0.0
            
            # Convert similarity (-1 to 1) to confidence score (0 to 1)
            score = (similarity + 1) / 2
            
            if score >= 0.9:
                reasoning = "Extremely high semantic similarity between query and content"
            elif score >= 0.7:
                reasoning = "High semantic similarity indicates good relevance"
            elif score >= 0.5:
                reasoning = "Moderate semantic similarity"
            else:
                reasoning = "Low semantic similarity may indicate poor relevance"
            
            return ConfidenceFactor(
                factor_type=FactorType.EMBEDDING_SIMILARITY,
                score=score,
                weight=self.factor_weights[FactorType.EMBEDDING_SIMILARITY],
                reasoning=reasoning,
                metadata={'cosine_similarity': similarity}
            )
            
        except Exception as e:
            logger.warning("Failed to calculate similarity factor", error=str(e))
            return ConfidenceFactor(
                factor_type=FactorType.EMBEDDING_SIMILARITY,
                score=0.5,
                weight=self.factor_weights[FactorType.EMBEDDING_SIMILARITY],
                reasoning="Could not calculate embedding similarity",
                metadata={'error': str(e)}
            )
    
    def _calculate_freshness_factor(self, metadata: Dict[str, Any]) -> ConfidenceFactor:
        """Calculate content freshness confidence factor."""
        try:
            # Try to get timestamp from metadata
            timestamp_str = metadata.get('timestamp') or metadata.get('created_at') or metadata.get('crawl_time')
            
            if timestamp_str:
                if isinstance(timestamp_str, str):
                    content_time = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                else:
                    content_time = timestamp_str
                    
                age_hours = (datetime.utcnow() - content_time).total_seconds() / 3600
                
                # Freshness score based on age
                if age_hours <= 1:
                    score = 1.0
                    reasoning = "Content is very fresh (less than 1 hour old)"
                elif age_hours <= 24:
                    score = 0.9
                    reasoning = "Content is fresh (less than 1 day old)"
                elif age_hours <= 168:  # 1 week
                    score = 0.7
                    reasoning = "Content is relatively fresh (less than 1 week old)"
                elif age_hours <= 720:  # 1 month
                    score = 0.5
                    reasoning = "Content is moderately fresh (less than 1 month old)"
                elif age_hours <= 8760:  # 1 year
                    score = 0.3
                    reasoning = "Content is aging (less than 1 year old)"
                else:
                    score = 0.1
                    reasoning = "Content is old (more than 1 year old)"
                
                metadata_info = {'age_hours': age_hours, 'content_time': content_time.isoformat()}
            else:
                score = 0.5
                reasoning = "No timestamp available, assuming moderate freshness"
                metadata_info = {'no_timestamp': True}
            
            return ConfidenceFactor(
                factor_type=FactorType.FRESHNESS,
                score=score,
                weight=self.factor_weights[FactorType.FRESHNESS],
                reasoning=reasoning,
                metadata=metadata_info
            )
            
        except Exception as e:
            logger.warning("Failed to calculate freshness factor", error=str(e))
            return ConfidenceFactor(
                factor_type=FactorType.FRESHNESS,
                score=0.5,
                weight=self.factor_weights[FactorType.FRESHNESS],
                reasoning="Could not determine content freshness",
                metadata={'error': str(e)}
            )
    
    def _calculate_domain_factor(self, domain_reputation: DomainReputation) -> ConfidenceFactor:
        """Calculate source domain confidence factor."""
        # Combine trust and authority scores
        combined_score = (domain_reputation.trust_score + domain_reputation.authority_score) / 2
        
        if combined_score >= 0.9:
            reasoning = f"Highly trusted domain ({domain_reputation.domain}) with excellent reputation"
        elif combined_score >= 0.7:
            reasoning = f"Trusted domain ({domain_reputation.domain}) with good reputation"
        elif combined_score >= 0.5:
            reasoning = f"Neutral domain ({domain_reputation.domain}) with average reputation"
        elif combined_score >= 0.3:
            reasoning = f"Questionable domain ({domain_reputation.domain}) with poor reputation"
        else:
            reasoning = f"Untrusted domain ({domain_reputation.domain}) with very poor reputation"
        
        return ConfidenceFactor(
            factor_type=FactorType.SOURCE_DOMAIN,
            score=combined_score,
            weight=self.factor_weights[FactorType.SOURCE_DOMAIN],
            reasoning=reasoning,
            metadata={
                'domain': domain_reputation.domain,
                'trust_score': domain_reputation.trust_score,
                'authority_score': domain_reputation.authority_score,
                'category': domain_reputation.reputation_category
            }
        )
    
    def _calculate_content_quality_factor(self, metrics: ContentMetrics, text: str) -> ConfidenceFactor:
        """Calculate content quality confidence factor."""
        quality_indicators = []
        score_components = []
        
        # Length quality
        if metrics.text_length >= 500:
            quality_indicators.append("good length")
            score_components.append(0.8)
        elif metrics.text_length >= 100:
            quality_indicators.append("adequate length")
            score_components.append(0.6)
        else:
            quality_indicators.append("short content")
            score_components.append(0.3)
        
        # Readability quality
        if metrics.readability_score >= 60:
            quality_indicators.append("good readability")
            score_components.append(0.7)
        elif metrics.readability_score >= 30:
            quality_indicators.append("moderate readability")
            score_components.append(0.5)
        else:
            quality_indicators.append("poor readability")
            score_components.append(0.3)
        
        # Structure quality
        if metrics.paragraph_count >= 3:
            quality_indicators.append("well-structured")
            score_components.append(0.7)
        elif metrics.paragraph_count >= 1:
            quality_indicators.append("basic structure")
            score_components.append(0.5)
        else:
            quality_indicators.append("poor structure")
            score_components.append(0.2)
        
        # Technical content quality
        if metrics.technical_terms_count >= 5:
            quality_indicators.append("technical depth")
            score_components.append(0.8)
        
        # Citation quality
        if metrics.citation_count >= 3:
            quality_indicators.append("well-cited")
            score_components.append(0.9)
        elif metrics.citation_count >= 1:
            quality_indicators.append("some citations")
            score_components.append(0.6)
        
        # Calculate overall quality score
        overall_score = sum(score_components) / len(score_components) if score_components else 0.5
        
        reasoning = f"Content quality assessment: {', '.join(quality_indicators)}"
        
        return ConfidenceFactor(
            factor_type=FactorType.CONTENT_QUALITY,
            score=overall_score,
            weight=self.factor_weights[FactorType.CONTENT_QUALITY],
            reasoning=reasoning,
            metadata={
                'quality_indicators': quality_indicators,
                'metrics': metrics.model_dump()
            }
        )
    
    def _calculate_extraction_method_factor(self, metadata: Dict[str, Any]) -> ConfidenceFactor:
        """Calculate extraction method confidence factor."""
        extraction_method = metadata.get('extraction_method', 'unknown')
        
        method_scores = {
            'trafilatura': 0.9,
            'newspaper3k': 0.8,
            'crawl4ai': 0.85,
            'hybrid': 0.9,
            'beautifulsoup': 0.6,
            'manual': 1.0,
            'api': 0.95
        }
        
        score = method_scores.get(extraction_method.lower(), 0.5)
        
        if score >= 0.9:
            reasoning = f"Excellent extraction method ({extraction_method}) with high reliability"
        elif score >= 0.7:
            reasoning = f"Good extraction method ({extraction_method}) with decent reliability"
        elif score >= 0.5:
            reasoning = f"Average extraction method ({extraction_method}) with moderate reliability"
        else:
            reasoning = f"Poor extraction method ({extraction_method}) with low reliability"
        
        return ConfidenceFactor(
            factor_type=FactorType.EXTRACTION_METHOD,
            score=score,
            weight=self.factor_weights[FactorType.EXTRACTION_METHOD],
            reasoning=reasoning,
            metadata={'extraction_method': extraction_method}
        )
    
    async def _calculate_cross_validation_factor(self, text: str, source: str) -> ConfidenceFactor:
        """Calculate cross-validation confidence factor."""
        # This would ideally compare with other sources
        # For now, use simple heuristics
        
        score = 0.5  # Default neutral score
        reasoning = "Cross-validation not available"
        
        # Simple validation based on content characteristics
        if len(text) > 1000:
            score += 0.1
        
        # Check for multiple sources or references in text
        if re.search(r'according to|reports suggest|studies show|research indicates', text, re.IGNORECASE):
            score += 0.2
            reasoning = "Content references multiple sources"
        
        score = min(1.0, score)
        
        return ConfidenceFactor(
            factor_type=FactorType.CROSS_VALIDATION,
            score=score,
            weight=self.factor_weights[FactorType.CROSS_VALIDATION],
            reasoning=reasoning,
            metadata={'validation_attempted': True}
        )
    
    def _calculate_temporal_consistency_factor(self, metadata: Dict[str, Any]) -> ConfidenceFactor:
        """Calculate temporal consistency confidence factor."""
        # Check if content timing is consistent with expectations
        score = 0.7  # Default good score
        reasoning = "No temporal inconsistencies detected"
        
        # Add more sophisticated temporal analysis here
        # For now, return default
        
        return ConfidenceFactor(
            factor_type=FactorType.TEMPORAL_CONSISTENCY,
            score=score,
            weight=self.factor_weights[FactorType.TEMPORAL_CONSISTENCY],
            reasoning=reasoning,
            metadata={'temporal_check': True}
        )
    
    def _calculate_authority_signals_factor(self, text: str, source: str) -> ConfidenceFactor:
        """Calculate authority signals confidence factor."""
        authority_indicators = []
        score_components = []
        
        # Check for authority signals in text
        if re.search(r'\b(PhD|Dr\.|Professor|researcher|scientist|expert)\b', text, re.IGNORECASE):
            authority_indicators.append("expert attribution")
            score_components.append(0.8)
        
        if re.search(r'\b(published|peer.?reviewed|journal|study|research)\b', text, re.IGNORECASE):
            authority_indicators.append("academic content")
            score_components.append(0.9)
        
        if re.search(r'\b(official|verified|confirmed|authenticated)\b', text, re.IGNORECASE):
            authority_indicators.append("official status")
            score_components.append(0.7)
        
        # Calculate score
        if score_components:
            overall_score = sum(score_components) / len(score_components)
            reasoning = f"Authority signals found: {', '.join(authority_indicators)}"
        else:
            overall_score = 0.5
            reasoning = "No clear authority signals detected"
        
        return ConfidenceFactor(
            factor_type=FactorType.AUTHORITY_SIGNALS,
            score=overall_score,
            weight=self.factor_weights[FactorType.AUTHORITY_SIGNALS],
            reasoning=reasoning,
            metadata={'authority_indicators': authority_indicators}
        )
    
    def _calculate_weighted_score(self, factors: List[ConfidenceFactor]) -> float:
        """Calculate weighted overall confidence score."""
        if not factors:
            return 0.0
        
        weighted_sum = sum(factor.score * factor.weight for factor in factors)
        total_weight = sum(factor.weight for factor in factors)
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def _determine_confidence_level(self, score: float) -> ConfidenceLevel:
        """Determine categorical confidence level from score."""
        if score >= 0.95:
            return ConfidenceLevel.ROCK_SOLID
        elif score >= 0.80:
            return ConfidenceLevel.HIGH
        elif score >= 0.60:
            return ConfidenceLevel.FUZZY
        else:
            return ConfidenceLevel.LOW
    
    async def _generate_reasoning(
        self,
        factors: List[ConfidenceFactor],
        overall_score: float,
        confidence_level: ConfidenceLevel,
        text: str,
        source: str,
        metadata: Dict[str, Any]
    ) -> Tuple[str, List[str], List[str]]:
        """Generate comprehensive reasoning, recommendations, and warnings."""
        
        # Generate AI reasoning if available
        if self.ai_agent:
            try:
                factor_summary = "\n".join([
                    f"- {factor.factor_type.value}: {factor.score:.2f} ({factor.reasoning})"
                    for factor in factors
                ])
                
                prompt = f"""
                Analyze the following confidence assessment for memory content:
                
                Overall Score: {overall_score:.2f}
                Confidence Level: {confidence_level.value}
                Source: {source}
                
                Individual Factors:
                {factor_summary}
                
                Content Preview: {text[:300]}...
                
                Provide:
                1. A clear summary of why this confidence level was assigned
                2. Specific recommendations for using this information
                3. Any warnings or concerns about reliability
                """
                
                ai_response = await self.ai_agent.run(prompt)
                ai_reasoning = ai_response.data
                
                # Parse AI response for structured output
                reasoning_lines = ai_reasoning.split('\n')
                reasoning = ai_reasoning
                recommendations = ["Use AI-generated recommendations"]
                warnings = ["Review AI-generated warnings"]
                
                return reasoning, recommendations, warnings
                
            except Exception as e:
                logger.warning("AI reasoning generation failed", error=str(e))
        
        # Fallback to rule-based reasoning
        reasoning_parts = []
        recommendations = []
        warnings = []
        
        # Overall assessment
        reasoning_parts.append(f"Overall confidence score: {overall_score:.2f} ({confidence_level.value})")
        
        # Top contributing factors
        sorted_factors = sorted(factors, key=lambda f: f.score * f.weight, reverse=True)
        top_factors = sorted_factors[:3]
        
        reasoning_parts.append("Key contributing factors:")
        for factor in top_factors:
            reasoning_parts.append(f"- {factor.factor_type.value}: {factor.reasoning}")
        
        # Generate recommendations based on confidence level
        if confidence_level == ConfidenceLevel.ROCK_SOLID:
            recommendations.extend([
                "Content is highly reliable for automated decision making",
                "Suitable for trading decisions and critical operations",
                "Can be used without additional verification"
            ])
        elif confidence_level == ConfidenceLevel.HIGH:
            recommendations.extend([
                "Content is generally reliable for most uses",
                "Consider additional verification for critical decisions",
                "Monitor for updates or conflicting information"
            ])
        elif confidence_level == ConfidenceLevel.FUZZY:
            recommendations.extend([
                "Use with caution and seek additional verification",
                "Cross-reference with other sources before acting",
                "Suitable for informational purposes with caveats"
            ])
        else:  # LOW
            recommendations.extend([
                "Do not use for important decisions without verification",
                "Seek alternative sources of information",
                "Consider this information unreliable"
            ])
            warnings.append("Low confidence - information may be unreliable")
        
        # Add specific warnings based on factors
        for factor in factors:
            if factor.score < 0.3:
                warnings.append(f"Poor {factor.factor_type.value}: {factor.reasoning}")
        
        return "\n".join(reasoning_parts), recommendations, warnings
    
    async def get_assessment_stats(self) -> Dict[str, Any]:
        """Get confidence assessment statistics."""
        return {
            **self.assessment_stats,
            'domain_cache_size': len(self.domain_cache),
            'factor_weights': self.factor_weights,
            'high_confidence_rate': (
                self.assessment_stats['high_confidence_count'] / 
                max(self.assessment_stats['total_assessments'], 1)
            )
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check of confidence agent."""
        health_status = {
            'status': 'healthy',
            'ai_agent_available': self.ai_agent is not None,
            'factor_weights_configured': bool(self.factor_weights),
            'domain_cache_size': len(self.domain_cache),
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Test AI agent if available
        if self.ai_agent:
            try:
                test_response = await self.ai_agent.run("Test prompt for health check")
                health_status['ai_agent_status'] = 'healthy'
            except Exception as e:
                health_status['ai_agent_status'] = 'unhealthy'
                health_status['ai_agent_error'] = str(e)
        
        return health_status


# Example usage
async def example_usage():
    """Example of using MemoryConfidenceAgent."""
    agent = MemoryConfidenceAgent()
    
    # Test content
    test_text = """
    Artificial intelligence (AI) is transforming the cryptocurrency trading landscape.
    Recent research from MIT shows that AI-powered trading algorithms can improve
    returns by up to 15% while reducing risk. This study, published in the Journal
    of Financial Technology, analyzed over 10,000 trading transactions across
    multiple cryptocurrency exchanges.
    """
    
    test_source = "https://example.com/ai-crypto-trading"
    test_metadata = {
        'timestamp': datetime.utcnow().isoformat(),
        'extraction_method': 'trafilatura'
    }
    
    # Assess confidence
    result = await agent.assess_confidence(
        text=test_text,
        source=test_source,
        metadata=test_metadata
    )
    
    print(f"Overall Score: {result.overall_score:.2f}")
    print(f"Confidence Level: {result.confidence_level.value}")
    print(f"Trading Safe: {result.is_trading_safe()}")
    print(f"Reasoning: {result.reasoning}")
    print(f"Recommendations: {result.recommendations}")
    print(f"Warnings: {result.warnings}")
    
    # Get statistics
    stats = await agent.get_assessment_stats()
    print(f"Assessment stats: {stats}")


if __name__ == "__main__":
    asyncio.run(example_usage())