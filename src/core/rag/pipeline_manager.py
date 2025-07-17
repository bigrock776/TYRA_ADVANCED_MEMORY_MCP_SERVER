"""
Integrated RAG Pipeline Manager.

This module provides a unified interface for the complete RAG pipeline,
integrating multi-modal processing, chunk linking, dynamic reranking,
intent detection, and hallucination detection into a cohesive system.
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from datetime import datetime, timezone

import structlog
from pydantic import BaseModel, Field, ConfigDict

from .multimodal import MultiModalProcessor, MultiModalContent, ModalityType
from .chunk_linking import ContextualChunkLinker, TextChunk, ChunkLinkingResult
from .dynamic_reranking import DynamicReranker, QueryIntent, QueryContext
from .intent_detector import IntentAwareRetriever, RetrievalContext, RetrievalResult
from .hallucination_detector import HallucinationDetector, HallucinationResult
from ..embeddings.embedder import Embedder
from ..memory.manager import MemoryManager
from ..graph.neo4j_client import Neo4jClient
from ..cache.redis_cache import RedisCache

logger = structlog.get_logger(__name__)


class PipelineStage(str, Enum):
    """Stages in the RAG pipeline."""
    INTENT_DETECTION = "intent_detection"
    MULTIMODAL_PROCESSING = "multimodal_processing"
    RETRIEVAL = "retrieval"
    CHUNK_LINKING = "chunk_linking"
    RERANKING = "reranking"
    HALLUCINATION_CHECK = "hallucination_check"
    RESPONSE_SYNTHESIS = "response_synthesis"


class PipelineMode(str, Enum):
    """RAG pipeline operation modes."""
    FAST = "fast"           # Skip some quality checks for speed
    STANDARD = "standard"   # Balanced speed and quality
    THOROUGH = "thorough"   # Maximum quality checks
    CUSTOM = "custom"       # User-defined pipeline


@dataclass
class PipelineConfig:
    """Configuration for RAG pipeline."""
    mode: PipelineMode = PipelineMode.STANDARD
    enabled_stages: List[PipelineStage] = field(default_factory=lambda: list(PipelineStage))
    
    # Performance settings
    max_retrieval_results: int = 50
    max_chunks_per_result: int = 10
    max_response_time_ms: int = 3000
    
    # Quality settings
    min_confidence_threshold: float = 0.7
    enable_hallucination_detection: bool = True
    enable_multi_modal: bool = True
    enable_chunk_linking: bool = True
    enable_reranking: bool = True
    
    # Caching
    cache_ttl_seconds: int = 300
    enable_pipeline_cache: bool = True


class PipelineMetrics(BaseModel):
    """Metrics for pipeline execution."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    total_time_ms: float = Field(..., description="Total pipeline execution time")
    stage_times: Dict[str, float] = Field(..., description="Time per pipeline stage")
    
    # Quality metrics
    retrieval_count: int = Field(..., description="Number of items retrieved")
    reranked_count: int = Field(..., description="Number of items after reranking")
    final_result_count: int = Field(..., description="Final number of results")
    average_confidence: float = Field(..., description="Average confidence score")
    
    # Stage success
    stages_completed: List[str] = Field(..., description="Successfully completed stages")
    stages_skipped: List[str] = Field(..., description="Skipped stages")
    stages_failed: List[str] = Field(..., description="Failed stages")
    
    # Resource usage
    cache_hits: int = Field(default=0, description="Number of cache hits")
    cache_misses: int = Field(default=0, description="Number of cache misses")


class RAGPipelineResult(BaseModel):
    """Result from complete RAG pipeline execution."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    query: str = Field(..., description="Original query")
    results: List[Dict[str, Any]] = Field(..., description="Final retrieved and processed results")
    
    # Pipeline execution info
    pipeline_mode: str = Field(..., description="Pipeline mode used")
    metrics: PipelineMetrics = Field(..., description="Execution metrics")
    
    # Quality assessment
    overall_confidence: float = Field(..., description="Overall confidence in results")
    hallucination_detected: bool = Field(default=False, description="Whether hallucination was detected")
    quality_score: float = Field(..., description="Overall quality score")
    
    # Metadata
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Pipeline execution time")
    user_id: Optional[str] = Field(default=None, description="User who executed pipeline")


class IntegratedRAGPipeline:
    """Integrated RAG Pipeline Manager."""
    
    def __init__(
        self,
        memory_manager: MemoryManager,
        embedder: Embedder,
        graph_client: Optional[Neo4jClient] = None,
        cache: Optional[RedisCache] = None
    ):
        self.memory_manager = memory_manager
        self.embedder = embedder
        self.graph_client = graph_client
        self.cache = cache
        
        # Initialize pipeline components
        self.multimodal_processor = MultiModalProcessor(embedder)
        self.chunk_linker = ContextualChunkLinker(embedder, graph_client)
        self.reranker = DynamicReranker(embedder, cache)
        self.intent_retriever = IntentAwareRetriever(memory_manager, embedder, graph_client)
        self.hallucination_detector = HallucinationDetector(embedder)
        
        # Pipeline state
        self.default_config = PipelineConfig()
        self.metrics_history: List[PipelineMetrics] = []
        
        logger.info("IntegratedRAGPipeline initialized")
    
    async def execute_pipeline(
        self,
        query: str,
        user_id: Optional[str] = None,
        config: Optional[PipelineConfig] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> RAGPipelineResult:
        """Execute the complete RAG pipeline."""
        start_time = time.time()
        
        if config is None:
            config = self.default_config
        
        # Initialize metrics tracking
        stage_times = {}
        stages_completed = []
        stages_skipped = []
        stages_failed = []
        cache_hits = 0
        cache_misses = 0
        
        try:
            # Check cache first
            if config.enable_pipeline_cache and self.cache:
                cache_key = f"rag_pipeline:{hash(query)}:{user_id}:{hash(str(config))}"
                cached_result = await self.cache.get(cache_key)
                if cached_result:
                    cache_hits += 1
                    logger.info("Pipeline result served from cache", query=query[:50])
                    return RAGPipelineResult.model_validate_json(cached_result)
                cache_misses += 1
            
            # Stage 1: Intent Detection
            stage_start = time.time()
            query_intent = None
            retrieval_context = None
            
            if PipelineStage.INTENT_DETECTION in config.enabled_stages:
                try:
                    retrieval_context = RetrievalContext(
                        query=query,
                        user_id=user_id or "anonymous",
                        context=context or {}
                    )
                    # Intent detection is built into the retrieval process
                    stages_completed.append(PipelineStage.INTENT_DETECTION.value)
                except Exception as e:
                    logger.error("Intent detection failed", error=str(e))
                    stages_failed.append(PipelineStage.INTENT_DETECTION.value)
            else:
                stages_skipped.append(PipelineStage.INTENT_DETECTION.value)
            
            stage_times[PipelineStage.INTENT_DETECTION.value] = (time.time() - stage_start) * 1000
            
            # Stage 2: Multi-modal Processing
            stage_start = time.time()
            processed_query = query
            
            if (PipelineStage.MULTIMODAL_PROCESSING in config.enabled_stages and 
                config.enable_multi_modal):
                try:
                    # Check if query contains multi-modal content references
                    multimodal_content = await self._extract_multimodal_content(query, context)
                    if multimodal_content:
                        processed_query = await self._enhance_query_with_multimodal(query, multimodal_content)
                    stages_completed.append(PipelineStage.MULTIMODAL_PROCESSING.value)
                except Exception as e:
                    logger.error("Multi-modal processing failed", error=str(e))
                    stages_failed.append(PipelineStage.MULTIMODAL_PROCESSING.value)
            else:
                stages_skipped.append(PipelineStage.MULTIMODAL_PROCESSING.value)
            
            stage_times[PipelineStage.MULTIMODAL_PROCESSING.value] = (time.time() - stage_start) * 1000
            
            # Stage 3: Retrieval
            stage_start = time.time()
            retrieval_results = []
            
            if PipelineStage.RETRIEVAL in config.enabled_stages:
                try:
                    if retrieval_context is None:
                        retrieval_context = RetrievalContext(
                            query=processed_query,
                            user_id=user_id or "anonymous",
                            context=context or {}
                        )
                    
                    retrieval_result = await self.intent_retriever.retrieve_with_intent(
                        retrieval_context,
                        max_results=config.max_retrieval_results
                    )
                    retrieval_results = retrieval_result.results
                    stages_completed.append(PipelineStage.RETRIEVAL.value)
                except Exception as e:
                    logger.error("Retrieval failed", error=str(e))
                    stages_failed.append(PipelineStage.RETRIEVAL.value)
            else:
                stages_skipped.append(PipelineStage.RETRIEVAL.value)
            
            stage_times[PipelineStage.RETRIEVAL.value] = (time.time() - stage_start) * 1000
            
            # Stage 4: Chunk Linking
            stage_start = time.time()
            linked_chunks = retrieval_results
            
            if (PipelineStage.CHUNK_LINKING in config.enabled_stages and 
                config.enable_chunk_linking and retrieval_results):
                try:
                    # Convert results to chunks for linking
                    chunks = []
                    for result in retrieval_results:
                        chunk = TextChunk(
                            chunk_id=result.get('id', ''),
                            content=result.get('content', ''),
                            source_id=result.get('memory_id', ''),
                            position=0,
                            metadata=result.get('metadata', {})
                        )
                        chunks.append(chunk)
                    
                    linking_result = await self.chunk_linker.link_chunks(chunks, processed_query)
                    # Update results with linking information
                    linked_chunks = await self._integrate_chunk_links(retrieval_results, linking_result)
                    stages_completed.append(PipelineStage.CHUNK_LINKING.value)
                except Exception as e:
                    logger.error("Chunk linking failed", error=str(e))
                    stages_failed.append(PipelineStage.CHUNK_LINKING.value)
            else:
                stages_skipped.append(PipelineStage.CHUNK_LINKING.value)
            
            stage_times[PipelineStage.CHUNK_LINKING.value] = (time.time() - stage_start) * 1000
            
            # Stage 5: Reranking
            stage_start = time.time()
            reranked_results = linked_chunks
            
            if (PipelineStage.RERANKING in config.enabled_stages and 
                config.enable_reranking and linked_chunks):
                try:
                    query_context = QueryContext(
                        query=processed_query,
                        user_id=user_id or "anonymous",
                        session_context=context or {}
                    )
                    
                    reranking_result = await self.reranker.rerank_results(
                        linked_chunks, query_context
                    )
                    reranked_results = reranking_result.reranked_results
                    stages_completed.append(PipelineStage.RERANKING.value)
                except Exception as e:
                    logger.error("Reranking failed", error=str(e))
                    stages_failed.append(PipelineStage.RERANKING.value)
            else:
                stages_skipped.append(PipelineStage.RERANKING.value)
            
            stage_times[PipelineStage.RERANKING.value] = (time.time() - stage_start) * 1000
            
            # Stage 6: Hallucination Detection
            stage_start = time.time()
            final_results = reranked_results
            hallucination_detected = False
            overall_confidence = 1.0
            
            if (PipelineStage.HALLUCINATION_CHECK in config.enabled_stages and 
                config.enable_hallucination_detection and reranked_results):
                try:
                    # Check for hallucinations in the results
                    hallucination_result = await self._check_hallucinations(
                        processed_query, reranked_results
                    )
                    hallucination_detected = hallucination_result.has_hallucination
                    overall_confidence = hallucination_result.confidence_score
                    
                    # Filter out low-confidence results
                    final_results = [
                        result for result in reranked_results
                        if result.get('confidence', 1.0) >= config.min_confidence_threshold
                    ]
                    stages_completed.append(PipelineStage.HALLUCINATION_CHECK.value)
                except Exception as e:
                    logger.error("Hallucination detection failed", error=str(e))
                    stages_failed.append(PipelineStage.HALLUCINATION_CHECK.value)
            else:
                stages_skipped.append(PipelineStage.HALLUCINATION_CHECK.value)
            
            stage_times[PipelineStage.HALLUCINATION_CHECK.value] = (time.time() - stage_start) * 1000
            
            # Calculate final metrics
            total_time = (time.time() - start_time) * 1000
            
            # Calculate quality metrics
            avg_confidence = np.mean([
                result.get('confidence', 0.0) for result in final_results
            ]) if final_results else 0.0
            
            quality_score = self._calculate_quality_score(
                len(stages_completed), len(stages_failed), avg_confidence, overall_confidence
            )
            
            # Create metrics
            metrics = PipelineMetrics(
                total_time_ms=total_time,
                stage_times=stage_times,
                retrieval_count=len(retrieval_results),
                reranked_count=len(reranked_results),
                final_result_count=len(final_results),
                average_confidence=avg_confidence,
                stages_completed=stages_completed,
                stages_skipped=stages_skipped,
                stages_failed=stages_failed,
                cache_hits=cache_hits,
                cache_misses=cache_misses
            )
            
            # Create final result
            pipeline_result = RAGPipelineResult(
                query=query,
                results=final_results,
                pipeline_mode=config.mode.value,
                metrics=metrics,
                overall_confidence=overall_confidence,
                hallucination_detected=hallucination_detected,
                quality_score=quality_score,
                user_id=user_id
            )
            
            # Cache result if enabled
            if config.enable_pipeline_cache and self.cache:
                await self.cache.set(
                    cache_key,
                    pipeline_result.model_dump_json(),
                    expire_seconds=config.cache_ttl_seconds
                )
            
            # Store metrics
            self.metrics_history.append(metrics)
            if len(self.metrics_history) > 1000:  # Keep last 1000 executions
                self.metrics_history = self.metrics_history[-1000:]
            
            logger.info(
                "RAG pipeline completed",
                query=query[:50],
                total_time_ms=total_time,
                result_count=len(final_results),
                quality_score=quality_score
            )
            
            return pipeline_result
            
        except Exception as e:
            logger.error("Pipeline execution failed", error=str(e), query=query[:50])
            
            # Return error result
            total_time = (time.time() - start_time) * 1000
            error_metrics = PipelineMetrics(
                total_time_ms=total_time,
                stage_times=stage_times,
                retrieval_count=0,
                reranked_count=0,
                final_result_count=0,
                average_confidence=0.0,
                stages_completed=stages_completed,
                stages_skipped=stages_skipped,
                stages_failed=stages_failed + ["pipeline_execution"],
                cache_hits=cache_hits,
                cache_misses=cache_misses
            )
            
            return RAGPipelineResult(
                query=query,
                results=[],
                pipeline_mode=config.mode.value,
                metrics=error_metrics,
                overall_confidence=0.0,
                hallucination_detected=False,
                quality_score=0.0,
                user_id=user_id
            )
    
    async def _extract_multimodal_content(
        self,
        query: str,
        context: Optional[Dict[str, Any]]
    ) -> Optional[List[MultiModalContent]]:
        """Extract multi-modal content from query or context."""
        try:
            multimodal_content = []
            
            # Check context for files or media
            if context and 'files' in context:
                for file_info in context['files']:
                    content = MultiModalContent(
                        content_id=file_info.get('id', ''),
                        modality=ModalityType.IMAGE if file_info.get('type', '').startswith('image') else ModalityType.TEXT,
                        content=file_info.get('content', ''),
                        metadata=file_info
                    )
                    multimodal_content.append(content)
            
            return multimodal_content if multimodal_content else None
            
        except Exception as e:
            logger.error("Failed to extract multimodal content", error=str(e))
            return None
    
    async def _enhance_query_with_multimodal(
        self,
        query: str,
        multimodal_content: List[MultiModalContent]
    ) -> str:
        """Enhance query with multimodal content information."""
        try:
            enhanced_query = query
            
            for content in multimodal_content:
                if content.modality == ModalityType.IMAGE:
                    # Process image and add description to query
                    image_result = await self.multimodal_processor.process_image(content)
                    if image_result.description:
                        enhanced_query += f" [Image context: {image_result.description}]"
                
                elif content.modality == ModalityType.TEXT:
                    # Add relevant text content
                    enhanced_query += f" [Document context: {content.content[:200]}...]"
            
            return enhanced_query
            
        except Exception as e:
            logger.error("Failed to enhance query with multimodal content", error=str(e))
            return query
    
    async def _integrate_chunk_links(
        self,
        results: List[Dict[str, Any]],
        linking_result: ChunkLinkingResult
    ) -> List[Dict[str, Any]]:
        """Integrate chunk linking information into results."""
        try:
            # Create mapping from chunk IDs to links
            link_map = {}
            for link in linking_result.links:
                if link.source_chunk_id not in link_map:
                    link_map[link.source_chunk_id] = []
                link_map[link.source_chunk_id].append(link)
            
            # Add link information to results
            enhanced_results = []
            for result in results:
                result_copy = result.copy()
                result_id = result.get('id', '')
                
                if result_id in link_map:
                    result_copy['chunk_links'] = [
                        {
                            'target_id': link.target_chunk_id,
                            'link_type': link.link_type.value,
                            'strength': link.strength,
                            'explanation': link.explanation
                        }
                        for link in link_map[result_id]
                    ]
                    result_copy['linking_confidence'] = np.mean([link.confidence for link in link_map[result_id]])
                
                enhanced_results.append(result_copy)
            
            return enhanced_results
            
        except Exception as e:
            logger.error("Failed to integrate chunk links", error=str(e))
            return results
    
    async def _check_hallucinations(
        self,
        query: str,
        results: List[Dict[str, Any]]
    ) -> HallucinationResult:
        """Check results for potential hallucinations."""
        try:
            # Combine results into a single text for hallucination check
            combined_text = "\n\n".join([
                result.get('content', '') for result in results
            ])
            
            # Use hallucination detector
            hallucination_result = await self.hallucination_detector.detect_hallucination(
                query, combined_text
            )
            
            return hallucination_result
            
        except Exception as e:
            logger.error("Hallucination detection failed", error=str(e))
            # Return safe default
            return HallucinationResult(
                has_hallucination=False,
                confidence_score=0.5,
                hallucination_types=[],
                explanation="Hallucination check failed",
                validated_claims=[],
                unsupported_claims=[],
                source_grounding_score=0.5
            )
    
    def _calculate_quality_score(
        self,
        completed_stages: int,
        failed_stages: int,
        avg_confidence: float,
        overall_confidence: float
    ) -> float:
        """Calculate overall quality score for pipeline execution."""
        # Stage completion factor (0.4 weight)
        total_stages = len(PipelineStage)
        stage_factor = (completed_stages - failed_stages) / total_stages
        stage_factor = max(0.0, min(1.0, stage_factor))
        
        # Confidence factor (0.6 weight)
        confidence_factor = (avg_confidence + overall_confidence) / 2
        
        # Combined quality score
        quality_score = (stage_factor * 0.4) + (confidence_factor * 0.6)
        
        return max(0.0, min(1.0, quality_score))
    
    async def get_pipeline_analytics(self) -> Dict[str, Any]:
        """Get analytics about pipeline performance."""
        if not self.metrics_history:
            return {
                "total_executions": 0,
                "avg_execution_time_ms": 0,
                "avg_quality_score": 0,
                "stage_success_rates": {},
                "cache_hit_rate": 0
            }
        
        total_executions = len(self.metrics_history)
        avg_time = np.mean([m.total_time_ms for m in self.metrics_history])
        
        # Calculate stage success rates
        stage_success_rates = {}
        for stage in PipelineStage:
            completed_count = sum(1 for m in self.metrics_history if stage.value in m.stages_completed)
            stage_success_rates[stage.value] = completed_count / total_executions
        
        # Cache statistics
        total_cache_requests = sum(m.cache_hits + m.cache_misses for m in self.metrics_history)
        total_cache_hits = sum(m.cache_hits for m in self.metrics_history)
        cache_hit_rate = total_cache_hits / max(total_cache_requests, 1)
        
        return {
            "total_executions": total_executions,
            "avg_execution_time_ms": avg_time,
            "stage_success_rates": stage_success_rates,
            "cache_hit_rate": cache_hit_rate,
            "avg_retrieval_count": np.mean([m.retrieval_count for m in self.metrics_history]),
            "avg_final_result_count": np.mean([m.final_result_count for m in self.metrics_history])
        }


# Export the main pipeline class
__all__ = [
    "IntegratedRAGPipeline",
    "RAGPipelineResult", 
    "PipelineConfig",
    "PipelineMetrics",
    "PipelineStage",
    "PipelineMode"
]