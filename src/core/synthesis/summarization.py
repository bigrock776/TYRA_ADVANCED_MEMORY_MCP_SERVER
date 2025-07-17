"""
AI-Powered Memory Summarization with Anti-Hallucination.

This module provides intelligent memory summarization using Pydantic AI for structured
outputs and multi-layer hallucination detection. Supports both extractive and
abstractive summarization with quality scoring and progressive refinement.
"""

import asyncio
from typing import List, Dict, Optional, Tuple, Set, Any, Union
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import numpy as np
from rouge_score import rouge_scorer
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import structlog
from pydantic import BaseModel, Field, field_validator, ConfigDict
from pydantic_ai import Agent, ValidationError as PydanticAIValidationError
import re
from collections import Counter
import hashlib

from ...models.memory import Memory
from ..embeddings.embedder import Embedder
from ..rag.hallucination import HallucinationDetector
from ..cache.redis_cache import RedisCache
from ..utils.config import settings
from ..clients.vllm_client import VLLMClient

logger = structlog.get_logger(__name__)


class SummarizationType(str, Enum):
    """Types of summarization strategies."""
    EXTRACTIVE = "extractive"  # Extract key sentences
    ABSTRACTIVE = "abstractive"  # Generate new summary
    HYBRID = "hybrid"  # Combine both approaches
    PROGRESSIVE = "progressive"  # Multi-stage refinement


class SummaryQuality(str, Enum):
    """Quality levels for summaries."""
    EXCELLENT = "excellent"  # ROUGE > 0.7, factually consistent
    GOOD = "good"  # ROUGE > 0.5, mostly consistent
    FAIR = "fair"  # ROUGE > 0.3, some inconsistencies
    POOR = "poor"  # ROUGE < 0.3, significant issues


@dataclass
class QualityMetrics:
    """Quality metrics for summary evaluation."""
    rouge_l_score: float
    rouge_1_score: float
    rouge_2_score: float
    factual_consistency: float
    hallucination_score: float
    coverage_score: float
    coherence_score: float
    overall_quality: SummaryQuality


class MemorySummary(BaseModel):
    """Validated memory summary structure using Pydantic AI."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    summary: str = Field(..., min_length=10, max_length=500, description="The summary text")
    key_points: List[str] = Field(..., min_items=1, max_items=10, description="Key points extracted")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence in summary accuracy")
    source_references: List[str] = Field(..., min_items=1, description="References to source memories")
    factual_claims: List[str] = Field(default_factory=list, description="Factual claims made")
    uncertainty_flags: List[str] = Field(default_factory=list, description="Areas of uncertainty")
    
    @field_validator('summary')
    @classmethod
    def validate_summary_grounding(cls, v, info):
        """Ensure summary is grounded in source material."""
        context = info.context
        if context and 'source_texts' in context:
            source_texts = context['source_texts']
            # Basic check: summary should contain words from source
            source_words = set(' '.join(source_texts).lower().split())
            summary_words = set(v.lower().split())
            overlap = len(source_words.intersection(summary_words)) / len(summary_words)
            if overlap < 0.3:  # At least 30% word overlap
                raise ValueError(f"Summary not sufficiently grounded in source: {overlap:.2%} overlap")
        return v
    
    @field_validator('key_points')
    @classmethod
    def validate_key_points_uniqueness(cls, v):
        """Ensure key points are unique and meaningful."""
        # Remove duplicates while preserving order
        seen = set()
        unique_points = []
        for point in v:
            point_lower = point.lower().strip()
            if point_lower not in seen and len(point) > 5:
                seen.add(point_lower)
                unique_points.append(point)
        return unique_points


class SummarizationEngine:
    """
    Advanced memory summarization engine with anti-hallucination capabilities.
    
    Features:
    - Extractive summarization using TF-IDF and TextRank
    - Abstractive summarization with vLLM and Pydantic AI
    - Multi-layer hallucination detection
    - Progressive summarization with quality refinement
    - Source grounding validation
    - ROUGE-based quality scoring
    """
    
    def __init__(
        self,
        embedder: Embedder,
        vllm_client: Optional[VLLMClient] = None,
        hallucination_detector: Optional[HallucinationDetector] = None,
        cache: Optional[RedisCache] = None,
        spacy_model: str = "en_core_web_sm"
    ):
        """
        Initialize the summarization engine.
        
        Args:
            embedder: Embedder for semantic analysis
            vllm_client: vLLM client for abstractive summarization
            hallucination_detector: Detector for hallucination checking
            cache: Redis cache for performance
            spacy_model: SpaCy model for NLP tasks
        """
        self.embedder = embedder
        self.vllm_client = vllm_client
        self.hallucination_detector = hallucination_detector
        self.cache = cache
        
        # Initialize NLP components
        try:
            self.nlp = spacy.load(spacy_model)
        except:
            logger.warning(f"SpaCy model {spacy_model} not found, downloading...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", spacy_model])
            self.nlp = spacy.load(spacy_model)
        
        # Initialize ROUGE scorer
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'], 
            use_stemmer=True
        )
        
        # Initialize local summarization models as fallback
        self._init_local_models()
        
        # Initialize Pydantic AI agent for structured summarization
        if self.vllm_client:
            self._init_pydantic_agent()
        
        logger.info("Initialized summarization engine")
    
    def _init_local_models(self):
        """Initialize local models for fallback summarization."""
        try:
            # Use a small local T5 or BART model
            model_name = settings.synthesis.summarization.local_model
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.local_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            
            # Create summarization pipeline
            self.local_summarizer = pipeline(
                "summarization",
                model=self.local_model,
                tokenizer=self.tokenizer,
                device=0 if torch.cuda.is_available() else -1
            )
            logger.info(f"Loaded local summarization model: {model_name}")
        except Exception as e:
            logger.warning(f"Failed to load local summarization model: {e}")
            self.local_summarizer = None
    
    def _init_pydantic_agent(self):
        """Initialize Pydantic AI agent for structured summarization."""
        try:
            self.summary_agent = Agent(
                model=self.vllm_client,
                result_type=MemorySummary,
                system_prompt="""You are a memory synthesis expert specializing in accurate summarization.
                
                Your responsibilities:
                1. Create concise, accurate summaries that preserve key information
                2. Extract the most important points from the memories
                3. Maintain factual consistency with source material
                4. Flag any areas of uncertainty or ambiguity
                5. Never add information not present in the source
                6. Provide clear references to source content
                
                Focus on accuracy over creativity. When uncertain, acknowledge it."""
            )
            logger.info("Initialized Pydantic AI summary agent")
        except Exception as e:
            logger.warning(f"Failed to initialize Pydantic AI agent: {e}")
            self.summary_agent = None
    
    async def summarize_memories(
        self,
        memories: List[Memory],
        summarization_type: SummarizationType = SummarizationType.HYBRID,
        max_length: int = 200,
        min_length: int = 50,
        quality_threshold: float = 0.7
    ) -> Tuple[MemorySummary, QualityMetrics]:
        """
        Summarize a collection of memories with anti-hallucination checks.
        
        Args:
            memories: List of memories to summarize
            summarization_type: Type of summarization to use
            max_length: Maximum summary length
            min_length: Minimum summary length
            quality_threshold: Minimum quality score required
            
        Returns:
            Tuple of (summary, quality_metrics)
        """
        if not memories:
            raise ValueError("No memories provided for summarization")
        
        start_time = datetime.utcnow()
        
        # Extract texts from memories
        source_texts = [m.content for m in memories]
        memory_ids = [m.id for m in memories]
        
        # Check cache first
        cache_key = self._generate_cache_key(source_texts, summarization_type)
        if self.cache:
            cached_result = await self.cache.get(f"summary:{cache_key}")
            if cached_result:
                logger.info("Returning cached summary")
                return cached_result
        
        # Generate summary based on type
        if summarization_type == SummarizationType.EXTRACTIVE:
            summary = await self._extractive_summarize(source_texts, max_length)
        elif summarization_type == SummarizationType.ABSTRACTIVE:
            summary = await self._abstractive_summarize(source_texts, max_length, min_length)
        elif summarization_type == SummarizationType.HYBRID:
            summary = await self._hybrid_summarize(source_texts, max_length, min_length)
        elif summarization_type == SummarizationType.PROGRESSIVE:
            summary = await self._progressive_summarize(source_texts, max_length, min_length)
        else:
            raise ValueError(f"Unknown summarization type: {summarization_type}")
        
        # Perform quality evaluation
        quality_metrics = await self._evaluate_summary_quality(
            summary, 
            source_texts,
            memory_ids
        )
        
        # If quality is below threshold, attempt improvement
        if quality_metrics.overall_quality in [SummaryQuality.POOR, SummaryQuality.FAIR]:
            if quality_metrics.factual_consistency < quality_threshold:
                logger.warning(
                    "Summary quality below threshold, attempting improvement",
                    quality=quality_metrics.overall_quality,
                    factual_consistency=quality_metrics.factual_consistency
                )
                summary = await self._improve_summary(summary, source_texts, quality_metrics)
                # Re-evaluate after improvement
                quality_metrics = await self._evaluate_summary_quality(
                    summary,
                    source_texts,
                    memory_ids
                )
        
        # Cache the result
        if self.cache:
            await self.cache.set(
                f"summary:{cache_key}",
                (summary, quality_metrics),
                ttl=3600  # 1 hour
            )
        
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        logger.info(
            "Summary generation complete",
            memories_count=len(memories),
            summary_length=len(summary.summary),
            quality=quality_metrics.overall_quality,
            processing_time_s=processing_time
        )
        
        return summary, quality_metrics
    
    def _generate_cache_key(self, texts: List[str], summary_type: SummarizationType) -> str:
        """Generate cache key for summary results."""
        content_hash = hashlib.sha256(''.join(sorted(texts)).encode()).hexdigest()[:16]
        return f"{summary_type.value}:{content_hash}"
    
    async def _extractive_summarize(
        self, 
        texts: List[str], 
        max_length: int
    ) -> MemorySummary:
        """Perform extractive summarization using TF-IDF and TextRank."""
        combined_text = ' '.join(texts)
        
        # Split into sentences
        doc = self.nlp(combined_text)
        sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 10]
        
        if not sentences:
            raise ValueError("No valid sentences found for summarization")
        
        # Calculate TF-IDF scores
        vectorizer = TfidfVectorizer(
            max_features=100,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        try:
            tfidf_matrix = vectorizer.fit_transform(sentences)
            
            # Calculate sentence scores
            sentence_scores = np.array(tfidf_matrix.sum(axis=1)).flatten()
            
            # Get top sentences
            top_indices = sentence_scores.argsort()[-5:][::-1]  # Top 5 sentences
            selected_sentences = [sentences[i] for i in top_indices]
            
            # Create summary
            summary_text = ' '.join(selected_sentences)
            
            # Trim to max length
            if len(summary_text) > max_length:
                summary_text = summary_text[:max_length].rsplit(' ', 1)[0] + '...'
            
            # Extract key points
            key_points = self._extract_key_points(texts, num_points=5)
            
            return MemorySummary(
                summary=summary_text,
                key_points=key_points,
                confidence_score=0.8,  # Extractive is generally reliable
                source_references=[f"Extracted from {len(texts)} memories"],
                factual_claims=[],
                uncertainty_flags=[]
            )
            
        except Exception as e:
            logger.error(f"Extractive summarization failed: {e}")
            # Fallback to simple truncation
            return MemorySummary(
                summary=combined_text[:max_length] + '...',
                key_points=["Failed to extract key points"],
                confidence_score=0.3,
                source_references=[f"Truncated from {len(texts)} memories"],
                factual_claims=[],
                uncertainty_flags=["Summarization error occurred"]
            )
    
    async def _abstractive_summarize(
        self,
        texts: List[str],
        max_length: int,
        min_length: int
    ) -> MemorySummary:
        """Perform abstractive summarization using vLLM with Pydantic AI."""
        combined_text = ' '.join(texts)
        
        # Use Pydantic AI agent if available
        if self.summary_agent and self.vllm_client:
            try:
                result = await self.summary_agent.run(
                    f"Summarize these memories into a coherent summary: {combined_text}",
                    deps={
                        "max_length": max_length,
                        "min_length": min_length,
                        "source_count": len(texts)
                    },
                    message_history=[],
                    model_settings={
                        "temperature": 0.3,  # Lower temperature for factual consistency
                        "max_tokens": max_length * 2
                    }
                )
                
                # Validate with context
                summary = result.data
                summary.model_validate(
                    summary.model_dump(),
                    context={'source_texts': texts}
                )
                
                return summary
                
            except (PydanticAIValidationError, Exception) as e:
                logger.warning(f"Pydantic AI summarization failed: {e}")
        
        # Fallback to local model
        if self.local_summarizer:
            try:
                result = self.local_summarizer(
                    combined_text,
                    max_length=max_length,
                    min_length=min_length,
                    do_sample=False  # Deterministic for consistency
                )
                
                summary_text = result[0]['summary_text']
                key_points = self._extract_key_points(texts, num_points=5)
                
                return MemorySummary(
                    summary=summary_text,
                    key_points=key_points,
                    confidence_score=0.7,
                    source_references=[f"Generated from {len(texts)} memories"],
                    factual_claims=self._extract_factual_claims(summary_text),
                    uncertainty_flags=[]
                )
                
            except Exception as e:
                logger.error(f"Local model summarization failed: {e}")
        
        # Final fallback to extractive
        return await self._extractive_summarize(texts, max_length)
    
    async def _hybrid_summarize(
        self,
        texts: List[str],
        max_length: int,
        min_length: int
    ) -> MemorySummary:
        """Combine extractive and abstractive approaches."""
        # First, get extractive summary for key content
        extractive = await self._extractive_summarize(texts, max_length * 2)
        
        # Then use abstractive on the extracted content
        if self.summary_agent or self.local_summarizer:
            refined_texts = [extractive.summary] + extractive.key_points
            abstractive = await self._abstractive_summarize(
                refined_texts,
                max_length,
                min_length
            )
            
            # Combine insights from both
            return MemorySummary(
                summary=abstractive.summary,
                key_points=list(set(extractive.key_points + abstractive.key_points))[:10],
                confidence_score=(extractive.confidence_score + abstractive.confidence_score) / 2,
                source_references=extractive.source_references + abstractive.source_references,
                factual_claims=abstractive.factual_claims,
                uncertainty_flags=abstractive.uncertainty_flags
            )
        
        return extractive
    
    async def _progressive_summarize(
        self,
        texts: List[str],
        max_length: int,
        min_length: int
    ) -> MemorySummary:
        """Multi-stage progressive summarization with refinement."""
        # Stage 1: Initial clustering and grouping
        clusters = await self._cluster_similar_content(texts)
        
        # Stage 2: Summarize each cluster
        cluster_summaries = []
        for cluster in clusters:
            summary = await self._extractive_summarize(cluster, max_length // len(clusters))
            cluster_summaries.append(summary.summary)
        
        # Stage 3: Final abstractive summarization
        final_summary = await self._abstractive_summarize(
            cluster_summaries,
            max_length,
            min_length
        )
        
        # Add progressive refinement metadata
        final_summary.source_references.append(
            f"Progressive summarization: {len(texts)} → {len(clusters)} clusters → final"
        )
        
        return final_summary
    
    async def _cluster_similar_content(self, texts: List[str]) -> List[List[str]]:
        """Cluster similar content for progressive summarization."""
        if len(texts) <= 3:
            return [texts]  # No clustering needed
        
        # Get embeddings for clustering
        embeddings = await self.embedder.embed_batch(texts)
        
        # Simple clustering using cosine similarity
        from sklearn.cluster import AgglomerativeClustering
        
        n_clusters = min(len(texts) // 3, 5)  # Max 5 clusters
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric='cosine',
            linkage='average'
        )
        
        labels = clustering.fit_predict(embeddings)
        
        # Group texts by cluster
        clusters = [[] for _ in range(n_clusters)]
        for i, label in enumerate(labels):
            clusters[label].append(texts[i])
        
        # Remove empty clusters
        return [c for c in clusters if c]
    
    def _extract_key_points(self, texts: List[str], num_points: int = 5) -> List[str]:
        """Extract key points from texts using NLP techniques."""
        combined_text = ' '.join(texts)
        doc = self.nlp(combined_text)
        
        # Extract noun phrases as potential key points
        noun_phrases = []
        for chunk in doc.noun_chunks:
            if len(chunk.text.split()) >= 2:  # Multi-word phrases
                noun_phrases.append(chunk.text)
        
        # Extract entities
        entities = [ent.text for ent in doc.ents if ent.label_ in ['PERSON', 'ORG', 'GPE', 'DATE']]
        
        # Combine and deduplicate
        all_points = list(set(noun_phrases + entities))
        
        # Score by frequency
        point_scores = Counter(all_points)
        
        # Get top points
        top_points = [point for point, _ in point_scores.most_common(num_points)]
        
        # Format as complete phrases
        formatted_points = []
        for point in top_points:
            # Find sentences containing this point
            for sent in doc.sents:
                if point.lower() in sent.text.lower():
                    formatted_points.append(sent.text.strip())
                    break
        
        return formatted_points[:num_points] if formatted_points else ["No key points extracted"]
    
    def _extract_factual_claims(self, text: str) -> List[str]:
        """Extract factual claims from text for validation."""
        doc = self.nlp(text)
        claims = []
        
        for sent in doc.sents:
            # Look for sentences with factual indicators
            if any(token.pos_ in ['NUM', 'PROPN'] for token in sent):
                # Contains numbers or proper nouns - likely factual
                claims.append(sent.text.strip())
            elif any(token.text.lower() in ['is', 'was', 'were', 'are', 'has', 'have'] for token in sent):
                # Contains state-of-being verbs
                claims.append(sent.text.strip())
        
        return claims[:10]  # Limit to 10 claims
    
    async def _evaluate_summary_quality(
        self,
        summary: MemorySummary,
        source_texts: List[str],
        memory_ids: List[str]
    ) -> QualityMetrics:
        """Evaluate summary quality using multiple metrics."""
        combined_source = ' '.join(source_texts)
        
        # Calculate ROUGE scores
        rouge_scores = self.rouge_scorer.score(combined_source, summary.summary)
        
        # Calculate factual consistency using hallucination detector
        factual_consistency = 1.0  # Default if no detector
        hallucination_score = 0.0
        
        if self.hallucination_detector:
            hallucination_result = await self.hallucination_detector.detect_hallucination(
                summary.summary,
                source_texts,
                memory_ids
            )
            factual_consistency = hallucination_result.grounding_score
            hallucination_score = hallucination_result.hallucination_score
        
        # Calculate coverage score (how much of source is covered)
        source_words = set(combined_source.lower().split())
        summary_words = set(summary.summary.lower().split())
        coverage_score = len(source_words.intersection(summary_words)) / len(source_words)
        
        # Calculate coherence score (simple readability metric)
        doc = self.nlp(summary.summary)
        avg_sent_length = np.mean([len(sent.text.split()) for sent in doc.sents])
        coherence_score = min(1.0, 1.0 - abs(avg_sent_length - 15) / 30)  # Optimal ~15 words/sentence
        
        # Determine overall quality
        avg_rouge = (
            rouge_scores['rouge1'].fmeasure + 
            rouge_scores['rouge2'].fmeasure + 
            rouge_scores['rougeL'].fmeasure
        ) / 3
        
        if avg_rouge > 0.7 and factual_consistency > 0.9:
            overall_quality = SummaryQuality.EXCELLENT
        elif avg_rouge > 0.5 and factual_consistency > 0.7:
            overall_quality = SummaryQuality.GOOD
        elif avg_rouge > 0.3 and factual_consistency > 0.5:
            overall_quality = SummaryQuality.FAIR
        else:
            overall_quality = SummaryQuality.POOR
        
        return QualityMetrics(
            rouge_l_score=rouge_scores['rougeL'].fmeasure,
            rouge_1_score=rouge_scores['rouge1'].fmeasure,
            rouge_2_score=rouge_scores['rouge2'].fmeasure,
            factual_consistency=factual_consistency,
            hallucination_score=hallucination_score,
            coverage_score=coverage_score,
            coherence_score=coherence_score,
            overall_quality=overall_quality
        )
    
    async def _improve_summary(
        self,
        summary: MemorySummary,
        source_texts: List[str],
        quality_metrics: QualityMetrics
    ) -> MemorySummary:
        """Improve summary based on quality metrics."""
        # Identify main issues
        issues = []
        if quality_metrics.factual_consistency < 0.7:
            issues.append("low factual consistency")
        if quality_metrics.coverage_score < 0.3:
            issues.append("poor coverage")
        if quality_metrics.coherence_score < 0.5:
            issues.append("poor coherence")
        
        logger.info(f"Improving summary due to: {', '.join(issues)}")
        
        # If hallucination is the issue, use more extractive approach
        if quality_metrics.hallucination_score > 0.3:
            return await self._extractive_summarize(source_texts, len(summary.summary))
        
        # If coverage is the issue, include more content
        if quality_metrics.coverage_score < 0.3:
            # Re-summarize with higher max length
            return await self._hybrid_summarize(
                source_texts,
                int(len(summary.summary) * 1.5),
                len(summary.summary)
            )
        
        # For other issues, try progressive summarization
        return await self._progressive_summarize(
            source_texts,
            len(summary.summary),
            len(summary.summary) // 2
        )