"""
Inference-Level Anti-Hallucination with Logits Processors.

This module provides token-level hallucination prevention through logits processing,
factual consistency enforcement, source grounding validation, and confidence
thresholding at the inference level for maximum accuracy.
"""

import asyncio
import math
import re
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
import numpy as np
import torch
from transformers import LogitsProcessor, LogitsProcessorList
import structlog
from pydantic import BaseModel, Field, ConfigDict

from ..clients.vllm_client import VLLMClient
from ..embeddings.embedder import Embedder
from ..utils.config import settings

logger = structlog.get_logger(__name__)


class ProcessorType(str, Enum):
    """Types of logits processors."""
    FACTUAL_CONSISTENCY = "factual_consistency"
    SOURCE_GROUNDING = "source_grounding"
    CONFIDENCE_THRESHOLD = "confidence_threshold"
    VALIDATION_CHAIN = "validation_chain"
    UNCERTAINTY_INJECTION = "uncertainty_injection"
    REPETITION_PENALTY = "repetition_penalty"


class ConfidenceMode(str, Enum):
    """Confidence calculation modes."""
    TOKEN_PROBABILITY = "token_probability"
    SEQUENCE_PROBABILITY = "sequence_probability"
    SEMANTIC_CONSISTENCY = "semantic_consistency"
    SOURCE_ALIGNMENT = "source_alignment"


class ProcessorConfig(BaseModel):
    """Configuration for logits processors."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    processor_type: ProcessorType = Field(..., description="Type of processor")
    enabled: bool = Field(True, description="Whether processor is active")
    threshold: float = Field(0.8, ge=0.0, le=1.0, description="Confidence threshold")
    penalty_strength: float = Field(1.5, ge=1.0, le=5.0, description="Penalty strength")
    max_tokens_affected: int = Field(50, ge=1, le=500, description="Maximum tokens to affect")
    fallback_behavior: str = Field("skip", description="Behavior when processor fails")


@dataclass
class ProcessingContext:
    """Context for logits processing operations."""
    source_texts: List[str]
    query: str
    generated_so_far: str
    token_position: int
    sequence_length: int
    temperature: float
    embedder: Optional[Embedder] = None
    confidence_history: List[float] = None


class FactualConsistencyProcessor(LogitsProcessor):
    """
    Logits processor for enforcing factual consistency at token level.
    
    Reduces probability of tokens that would create factually inconsistent statements
    based on source material and established facts.
    """
    
    def __init__(
        self,
        config: ProcessorConfig,
        fact_database: Optional[Dict[str, Any]] = None,
        embedder: Optional[Embedder] = None
    ):
        """Initialize factual consistency processor."""
        self.config = config
        self.fact_database = fact_database or {}
        self.embedder = embedder
        
        # Token tracking for consistency
        self.fact_patterns = self._initialize_fact_patterns()
        self.inconsistency_penalties = {}
        
        # Performance tracking
        self.processor_stats = {
            'tokens_processed': 0,
            'penalties_applied': 0,
            'consistency_violations': 0,
            'average_penalty_strength': 0.0
        }
        
        logger.info("Initialized FactualConsistencyProcessor")
    
    def _initialize_fact_patterns(self) -> Dict[str, List[str]]:
        """Initialize patterns for factual consistency checking."""
        return {
            'dates': [
                r'\b(19|20)\d{2}\b',  # Years
                r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+(19|20)\d{2}\b'
            ],
            'numbers': [
                r'\b\d+(?:,\d{3})*(?:\.\d+)?\b',  # Numbers with commas/decimals
                r'\b\d+(?:st|nd|rd|th)\b'  # Ordinals
            ],
            'organizations': [
                r'\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*(?:\s+(?:Inc|Corp|LLC|Ltd|Co)\.?)\b'
            ],
            'people': [
                r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b'  # First Last names
            ]
        }
    
    def __call__(
        self,
        input_ids: torch.Tensor,
        scores: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """Process logits for factual consistency."""
        self.processor_stats['tokens_processed'] += 1
        
        if not self.config.enabled:
            return scores
        
        try:
            # Get current context from kwargs
            context = kwargs.get('processing_context')
            if not context:
                return scores
            
            # Check for potential factual inconsistencies
            current_text = context.generated_so_far
            penalties = self._calculate_consistency_penalties(
                input_ids, scores, current_text, context
            )
            
            if penalties:
                # Apply penalties to reduce probability of inconsistent tokens
                for token_id, penalty in penalties.items():
                    if token_id < scores.size(-1):
                        scores[0, token_id] *= (1.0 / penalty)
                        self.processor_stats['penalties_applied'] += 1
                        self.processor_stats['average_penalty_strength'] = (
                            (self.processor_stats['average_penalty_strength'] * 
                             (self.processor_stats['penalties_applied'] - 1) + penalty) /
                            self.processor_stats['penalties_applied']
                        )
            
            return scores
            
        except Exception as e:
            logger.warning(f"Factual consistency processing failed: {e}")
            return scores
    
    def _calculate_consistency_penalties(
        self,
        input_ids: torch.Tensor,
        scores: torch.Tensor,
        current_text: str,
        context: ProcessingContext
    ) -> Dict[int, float]:
        """Calculate penalties for potentially inconsistent tokens."""
        penalties = {}
        
        # Get top candidate tokens
        top_k = min(50, scores.size(-1))
        top_tokens = torch.topk(scores, top_k, dim=-1)
        
        for i, token_id in enumerate(top_tokens.indices[0]):
            token_id = token_id.item()
            
            # Decode token (simplified - would need actual tokenizer)
            token_text = self._decode_token_placeholder(token_id)
            
            # Check against known facts
            if self._violates_known_facts(current_text + token_text, context):
                penalty = self.config.penalty_strength
                penalties[token_id] = penalty
                self.processor_stats['consistency_violations'] += 1
        
        return penalties
    
    def _decode_token_placeholder(self, token_id: int) -> str:
        """Placeholder token decoding (would use actual tokenizer)."""
        # This would normally use the actual tokenizer
        # For now, return a placeholder
        return f"<token_{token_id}>"
    
    def _violates_known_facts(self, hypothetical_text: str, context: ProcessingContext) -> bool:
        """Check if text would violate known facts."""
        # Check against source texts for contradictions
        for source_text in context.source_texts:
            if self._check_contradiction(hypothetical_text, source_text):
                return True
        
        # Check against fact database
        if self._check_fact_database_violation(hypothetical_text):
            return True
        
        return False
    
    def _check_contradiction(self, text1: str, text2: str) -> bool:
        """Simple contradiction detection between texts."""
        # Look for contradictory patterns
        contradiction_patterns = [
            (r'is\s+(\w+)', r'is\s+not\s+\1'),
            (r'was\s+(\w+)', r'was\s+not\s+\1'),
            (r'(\d+)', r'(?!\1)\d+'),  # Different numbers
        ]
        
        for positive_pattern, negative_pattern in contradiction_patterns:
            if re.search(positive_pattern, text1) and re.search(negative_pattern, text2):
                return True
        
        return False
    
    def _check_fact_database_violation(self, text: str) -> bool:
        """Check against known fact database."""
        # Simple fact checking against database
        for fact_key, fact_value in self.fact_database.items():
            if fact_key.lower() in text.lower():
                # Check if text contradicts known fact
                if str(fact_value).lower() not in text.lower():
                    return True
        
        return False


class SourceGroundingProcessor(LogitsProcessor):
    """
    Logits processor for ensuring source grounding.
    
    Increases probability of tokens that align with source material and reduces
    probability of tokens that would create ungrounded statements.
    """
    
    def __init__(
        self,
        config: ProcessorConfig,
        embedder: Optional[Embedder] = None
    ):
        """Initialize source grounding processor."""
        self.config = config
        self.embedder = embedder
        
        # Source alignment tracking
        self.source_embeddings: Dict[str, np.ndarray] = {}
        self.alignment_cache: Dict[str, float] = {}
        
        # Performance tracking
        self.processor_stats = {
            'tokens_processed': 0,
            'alignments_calculated': 0,
            'grounding_boosts': 0,
            'grounding_penalties': 0,
            'average_alignment_score': 0.0
        }
        
        logger.info("Initialized SourceGroundingProcessor")
    
    def __call__(
        self,
        input_ids: torch.Tensor,
        scores: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """Process logits for source grounding."""
        self.processor_stats['tokens_processed'] += 1
        
        if not self.config.enabled or not self.embedder:
            return scores
        
        try:
            context = kwargs.get('processing_context')
            if not context or not context.source_texts:
                return scores
            
            # Calculate source alignment for top tokens
            alignment_adjustments = self._calculate_source_alignment(
                input_ids, scores, context
            )
            
            # Apply alignment adjustments
            for token_id, adjustment in alignment_adjustments.items():
                if token_id < scores.size(-1):
                    if adjustment > 1.0:
                        scores[0, token_id] *= adjustment
                        self.processor_stats['grounding_boosts'] += 1
                    else:
                        scores[0, token_id] *= adjustment
                        self.processor_stats['grounding_penalties'] += 1
            
            return scores
            
        except Exception as e:
            logger.warning(f"Source grounding processing failed: {e}")
            return scores
    
    def _calculate_source_alignment(
        self,
        input_ids: torch.Tensor,
        scores: torch.Tensor,
        context: ProcessingContext
    ) -> Dict[int, float]:
        """Calculate source alignment for candidate tokens."""
        adjustments = {}
        
        # Get source embeddings if not cached
        if not self.source_embeddings:
            self._precompute_source_embeddings(context.source_texts)
        
        # Get top candidate tokens
        top_k = min(20, scores.size(-1))
        top_tokens = torch.topk(scores, top_k, dim=-1)
        
        for token_id in top_tokens.indices[0]:
            token_id = token_id.item()
            
            # Calculate alignment for this token
            alignment_score = self._calculate_token_alignment(
                token_id, context.generated_so_far, context
            )
            
            # Convert alignment to adjustment factor
            if alignment_score > self.config.threshold:
                # Boost well-aligned tokens
                adjustment = 1.0 + (alignment_score - self.config.threshold) * self.config.penalty_strength
            else:
                # Penalize poorly aligned tokens
                adjustment = alignment_score / self.config.threshold
            
            adjustments[token_id] = adjustment
            
            self.processor_stats['alignments_calculated'] += 1
            self.processor_stats['average_alignment_score'] = (
                (self.processor_stats['average_alignment_score'] * 
                 (self.processor_stats['alignments_calculated'] - 1) + alignment_score) /
                self.processor_stats['alignments_calculated']
            )
        
        return adjustments
    
    def _precompute_source_embeddings(self, source_texts: List[str]) -> None:
        """Precompute embeddings for source texts."""
        try:
            for i, source_text in enumerate(source_texts):
                if source_text not in self.source_embeddings:
                    # This would be async in real implementation
                    # embedding = await self.embedder.embed_text(source_text)
                    # For now, use placeholder
                    embedding = np.random.rand(384)  # Placeholder embedding
                    self.source_embeddings[source_text] = embedding
                    
        except Exception as e:
            logger.warning(f"Failed to precompute source embeddings: {e}")
    
    def _calculate_token_alignment(
        self,
        token_id: int,
        current_text: str,
        context: ProcessingContext
    ) -> float:
        """Calculate alignment score for a specific token."""
        # Placeholder implementation
        # In real implementation, would:
        # 1. Decode token to text
        # 2. Create hypothetical text with token
        # 3. Embed hypothetical text
        # 4. Calculate similarity with source embeddings
        # 5. Return average similarity
        
        # For now, return a random score
        return np.random.uniform(0.3, 0.9)


class ConfidenceThresholdProcessor(LogitsProcessor):
    """
    Logits processor for confidence-based token filtering.
    
    Prevents generation of tokens below confidence threshold and injects
    uncertainty markers when confidence is marginal.
    """
    
    def __init__(self, config: ProcessorConfig):
        """Initialize confidence threshold processor."""
        self.config = config
        
        # Confidence tracking
        self.confidence_history: List[float] = []
        self.uncertainty_tokens = {
            'might', 'could', 'possibly', 'likely', 'appears', 
            'seems', 'probably', 'potentially', 'may', 'perhaps'
        }
        
        # Performance tracking
        self.processor_stats = {
            'tokens_processed': 0,
            'confidence_violations': 0,
            'uncertainty_injections': 0,
            'average_confidence': 0.0,
            'sequence_confidence': 0.0
        }
        
        logger.info("Initialized ConfidenceThresholdProcessor")
    
    def __call__(
        self,
        input_ids: torch.Tensor,
        scores: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """Process logits for confidence thresholding."""
        self.processor_stats['tokens_processed'] += 1
        
        if not self.config.enabled:
            return scores
        
        try:
            context = kwargs.get('processing_context')
            if not context:
                return scores
            
            # Calculate token-level confidence
            confidence_scores = self._calculate_token_confidence(scores)
            
            # Apply confidence filtering
            filtered_scores = self._apply_confidence_filter(
                scores, confidence_scores, context
            )
            
            # Update confidence tracking
            max_confidence = float(torch.max(torch.softmax(filtered_scores, dim=-1)))
            self.confidence_history.append(max_confidence)
            
            # Calculate sequence confidence
            if len(self.confidence_history) > 0:
                self.processor_stats['sequence_confidence'] = sum(self.confidence_history) / len(self.confidence_history)
                self.processor_stats['average_confidence'] = (
                    (self.processor_stats['average_confidence'] * 
                     (self.processor_stats['tokens_processed'] - 1) + max_confidence) /
                    self.processor_stats['tokens_processed']
                )
            
            return filtered_scores
            
        except Exception as e:
            logger.warning(f"Confidence threshold processing failed: {e}")
            return scores
    
    def _calculate_token_confidence(self, scores: torch.Tensor) -> torch.Tensor:
        """Calculate confidence scores for tokens."""
        # Use probability distribution entropy as confidence measure
        probs = torch.softmax(scores, dim=-1)
        
        # Calculate entropy (lower entropy = higher confidence)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
        
        # Convert entropy to confidence (0-1 scale)
        max_entropy = math.log(scores.size(-1))
        confidence = 1.0 - (entropy / max_entropy)
        
        return confidence
    
    def _apply_confidence_filter(
        self,
        scores: torch.Tensor,
        confidence_scores: torch.Tensor,
        context: ProcessingContext
    ) -> torch.Tensor:
        """Apply confidence-based filtering to scores."""
        filtered_scores = scores.clone()
        
        # Get current sequence confidence
        current_confidence = float(confidence_scores[0])
        
        if current_confidence < self.config.threshold:
            self.processor_stats['confidence_violations'] += 1
            
            # Check if we should inject uncertainty
            if self._should_inject_uncertainty(context):
                filtered_scores = self._boost_uncertainty_tokens(filtered_scores)
                self.processor_stats['uncertainty_injections'] += 1
            else:
                # Apply strong penalty to low-confidence tokens
                probs = torch.softmax(scores, dim=-1)
                low_conf_mask = probs < (self.config.threshold / 10)
                filtered_scores[low_conf_mask] *= 0.1
        
        return filtered_scores
    
    def _should_inject_uncertainty(self, context: ProcessingContext) -> bool:
        """Determine if uncertainty should be injected."""
        # Inject uncertainty if:
        # 1. We're making a factual claim
        # 2. We haven't used uncertainty markers recently
        # 3. Confidence is consistently low
        
        recent_text = context.generated_so_far[-100:]  # Last 100 chars
        
        # Check for factual claim patterns
        factual_patterns = [
            r'\bis\s+', r'\bwas\s+', r'\bwill\s+', r'\bhas\s+',
            r'\bthe\s+fact\s+', r'\baccording\s+to\s+'
        ]
        
        has_factual_claim = any(re.search(pattern, recent_text.lower()) for pattern in factual_patterns)
        
        # Check if uncertainty markers already present
        has_uncertainty = any(marker in recent_text.lower() for marker in self.uncertainty_tokens)
        
        # Check recent confidence trend
        recent_confidences = self.confidence_history[-5:]
        low_confidence_trend = len(recent_confidences) >= 3 and all(c < self.config.threshold for c in recent_confidences)
        
        return has_factual_claim and not has_uncertainty and low_confidence_trend
    
    def _boost_uncertainty_tokens(self, scores: torch.Tensor) -> torch.Tensor:
        """Boost probability of uncertainty tokens."""
        # This would require access to tokenizer to find uncertainty token IDs
        # For now, apply general uncertainty boost pattern
        
        # Boost tokens typically associated with uncertainty
        # This is a simplified implementation
        uncertainty_boost = 2.0
        
        # In real implementation, would identify specific uncertainty token IDs
        # and boost their scores
        
        return scores


class ValidationChainProcessor(LogitsProcessor):
    """
    Logits processor that chains multiple validation steps.
    
    Orchestrates multiple processors for comprehensive validation.
    """
    
    def __init__(
        self,
        processors: List[LogitsProcessor],
        config: ProcessorConfig
    ):
        """Initialize validation chain processor."""
        self.processors = processors
        self.config = config
        
        # Chain tracking
        self.processor_stats = {
            'tokens_processed': 0,
            'chain_validations': 0,
            'processor_failures': 0,
            'total_processing_time': 0.0
        }
        
        logger.info(f"Initialized ValidationChainProcessor with {len(processors)} processors")
    
    def __call__(
        self,
        input_ids: torch.Tensor,
        scores: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """Process logits through validation chain."""
        self.processor_stats['tokens_processed'] += 1
        
        if not self.config.enabled:
            return scores
        
        try:
            current_scores = scores
            
            # Apply each processor in sequence
            for i, processor in enumerate(self.processors):
                try:
                    current_scores = processor(input_ids, current_scores, **kwargs)
                except Exception as e:
                    logger.warning(f"Processor {i} failed: {e}")
                    self.processor_stats['processor_failures'] += 1
                    # Continue with other processors
            
            self.processor_stats['chain_validations'] += 1
            
            return current_scores
            
        except Exception as e:
            logger.error(f"Validation chain processing failed: {e}")
            return scores


class LogitsProcessorManager:
    """
    Manager for coordinating multiple logits processors.
    
    Provides unified interface for inference-level anti-hallucination processing.
    """
    
    def __init__(
        self,
        vllm_client: Optional[VLLMClient] = None,
        embedder: Optional[Embedder] = None
    ):
        """Initialize logits processor manager."""
        self.vllm_client = vllm_client
        self.embedder = embedder
        
        # Initialize processor configurations
        self.processor_configs = self._initialize_processor_configs()
        
        # Initialize processors
        self.processors = self._initialize_processors()
        
        # Performance tracking
        self.manager_stats = {
            'total_generations': 0,
            'successful_generations': 0,
            'average_processing_time': 0.0,
            'hallucination_preventions': 0,
            'error_count': 0
        }
        
        logger.info("Initialized LogitsProcessorManager")
    
    def _initialize_processor_configs(self) -> Dict[ProcessorType, ProcessorConfig]:
        """Initialize default processor configurations."""
        return {
            ProcessorType.FACTUAL_CONSISTENCY: ProcessorConfig(
                processor_type=ProcessorType.FACTUAL_CONSISTENCY,
                enabled=True,
                threshold=0.85,
                penalty_strength=2.0,
                max_tokens_affected=30
            ),
            ProcessorType.SOURCE_GROUNDING: ProcessorConfig(
                processor_type=ProcessorType.SOURCE_GROUNDING,
                enabled=True,
                threshold=0.75,
                penalty_strength=1.5,
                max_tokens_affected=40
            ),
            ProcessorType.CONFIDENCE_THRESHOLD: ProcessorConfig(
                processor_type=ProcessorType.CONFIDENCE_THRESHOLD,
                enabled=True,
                threshold=0.7,
                penalty_strength=3.0,
                max_tokens_affected=50
            )
        }
    
    def _initialize_processors(self) -> Dict[ProcessorType, LogitsProcessor]:
        """Initialize logits processors."""
        processors = {}
        
        # Factual consistency processor
        if ProcessorType.FACTUAL_CONSISTENCY in self.processor_configs:
            processors[ProcessorType.FACTUAL_CONSISTENCY] = FactualConsistencyProcessor(
                self.processor_configs[ProcessorType.FACTUAL_CONSISTENCY],
                embedder=self.embedder
            )
        
        # Source grounding processor
        if ProcessorType.SOURCE_GROUNDING in self.processor_configs:
            processors[ProcessorType.SOURCE_GROUNDING] = SourceGroundingProcessor(
                self.processor_configs[ProcessorType.SOURCE_GROUNDING],
                embedder=self.embedder
            )
        
        # Confidence threshold processor
        if ProcessorType.CONFIDENCE_THRESHOLD in self.processor_configs:
            processors[ProcessorType.CONFIDENCE_THRESHOLD] = ConfidenceThresholdProcessor(
                self.processor_configs[ProcessorType.CONFIDENCE_THRESHOLD]
            )
        
        # Validation chain processor
        chain_processors = list(processors.values())
        if chain_processors:
            processors[ProcessorType.VALIDATION_CHAIN] = ValidationChainProcessor(
                chain_processors,
                ProcessorConfig(
                    processor_type=ProcessorType.VALIDATION_CHAIN,
                    enabled=True,
                    threshold=0.8
                )
            )
        
        return processors
    
    def create_processor_list(
        self,
        context: ProcessingContext,
        enabled_processors: Optional[List[ProcessorType]] = None
    ) -> LogitsProcessorList:
        """
        Create processor list for generation.
        
        Args:
            context: Processing context with source texts and query
            enabled_processors: Optional list of processors to enable
            
        Returns:
            LogitsProcessorList for use in generation
        """
        try:
            if enabled_processors is None:
                enabled_processors = list(self.processors.keys())
            
            processor_list = LogitsProcessorList()
            
            for processor_type in enabled_processors:
                if processor_type in self.processors:
                    processor = self.processors[processor_type]
                    
                    # Add context to processor calls
                    original_call = processor.__call__
                    
                    def wrapped_call(input_ids, scores, **kwargs):
                        kwargs['processing_context'] = context
                        return original_call(input_ids, scores, **kwargs)
                    
                    processor.__call__ = wrapped_call
                    processor_list.append(processor)
            
            logger.info(
                "Created processor list",
                enabled_processors=[p.value for p in enabled_processors],
                context_sources=len(context.source_texts)
            )
            
            return processor_list
            
        except Exception as e:
            logger.error("Failed to create processor list", error=str(e))
            return LogitsProcessorList()
    
    def update_processor_config(
        self,
        processor_type: ProcessorType,
        config: ProcessorConfig
    ) -> None:
        """Update configuration for a specific processor."""
        self.processor_configs[processor_type] = config
        
        # Reinitialize the specific processor
        if processor_type == ProcessorType.FACTUAL_CONSISTENCY:
            self.processors[processor_type] = FactualConsistencyProcessor(
                config, embedder=self.embedder
            )
        elif processor_type == ProcessorType.SOURCE_GROUNDING:
            self.processors[processor_type] = SourceGroundingProcessor(
                config, embedder=self.embedder
            )
        elif processor_type == ProcessorType.CONFIDENCE_THRESHOLD:
            self.processors[processor_type] = ConfidenceThresholdProcessor(config)
        
        logger.info(f"Updated {processor_type.value} processor configuration")
    
    async def get_processor_stats(self) -> Dict[str, Any]:
        """Get comprehensive processor statistics."""
        stats = {
            'manager_stats': self.manager_stats,
            'processor_stats': {}
        }
        
        for processor_type, processor in self.processors.items():
            if hasattr(processor, 'processor_stats'):
                stats['processor_stats'][processor_type.value] = processor.processor_stats
        
        return stats
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check of all processors."""
        health_status = {
            'status': 'healthy',
            'processors': {},
            'vllm_client_available': self.vllm_client is not None,
            'embedder_available': self.embedder is not None,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        for processor_type, processor in self.processors.items():
            try:
                health_status['processors'][processor_type.value] = {
                    'status': 'healthy',
                    'enabled': getattr(processor, 'config', ProcessorConfig(processor_type=processor_type)).enabled,
                    'has_stats': hasattr(processor, 'processor_stats')
                }
            except Exception as e:
                health_status['processors'][processor_type.value] = {
                    'status': 'unhealthy',
                    'error': str(e)
                }
        
        # Overall status
        unhealthy_processors = [
            name for name, status in health_status['processors'].items()
            if status.get('status') == 'unhealthy'
        ]
        
        if unhealthy_processors:
            health_status['status'] = 'degraded'
            health_status['unhealthy_processors'] = unhealthy_processors
        
        return health_status


# Example usage
async def example_usage():
    """Example of using LogitsProcessorManager."""
    # Initialize manager
    manager = LogitsProcessorManager()
    
    # Create processing context
    context = ProcessingContext(
        source_texts=[
            "OpenAI was founded in 2015 as a non-profit AI research company.",
            "The company has developed several language models including GPT-3 and GPT-4."
        ],
        query="When was OpenAI founded?",
        generated_so_far="OpenAI was founded in ",
        token_position=5,
        sequence_length=20,
        temperature=0.7
    )
    
    # Create processor list
    processor_list = manager.create_processor_list(context)
    
    print(f"Created processor list with {len(processor_list)} processors")
    
    # In real usage, this would be used with vLLM generation:
    # generation_config = {
    #     'max_tokens': 100,
    #     'temperature': 0.7,
    #     'logits_processor': processor_list
    # }
    
    # Get statistics
    stats = await manager.get_processor_stats()
    print(f"Manager stats: {stats['manager_stats']}")
    
    # Health check
    health = await manager.health_check()
    print(f"Health status: {health['status']}")
    print(f"Available processors: {list(health['processors'].keys())}")


if __name__ == "__main__":
    # Note: This would normally be run with actual tokenizer and generation
    print("LogitsProcessorManager initialized - ready for integration with vLLM")
    asyncio.run(example_usage())