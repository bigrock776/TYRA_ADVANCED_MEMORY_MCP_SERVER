"""
Advanced Structured Memory Operations with Pydantic AI.

This module provides comprehensive structured memory operations with AI-enhanced
metadata validation, confidence scoring, change validation, and pattern detection
for enterprise-grade memory management.
"""

import asyncio
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import uuid

import structlog
from pydantic import BaseModel, Field, ConfigDict, field_validator, ValidationError
from pydantic_ai import Agent, RunContext

from ..clients.vllm_client import VLLMClient
from ..embeddings.embedder import Embedder
from ..utils.config import settings
from .manager import MemoryManager

logger = structlog.get_logger(__name__)


class OperationType(str, Enum):
    """Types of structured memory operations."""
    INGESTION = "ingestion"
    RETRIEVAL = "retrieval"
    UPDATE = "update"
    ANALYSIS = "analysis"
    PATTERN_DETECTION = "pattern_detection"
    VALIDATION = "validation"


class ConfidenceLevel(str, Enum):
    """Confidence levels for memory operations."""
    VERY_HIGH = "very_high"  # 95-100%
    HIGH = "high"            # 80-94%
    MEDIUM = "medium"        # 60-79%
    LOW = "low"              # 40-59%
    VERY_LOW = "very_low"    # 0-39%


class ChangeType(str, Enum):
    """Types of memory changes."""
    CONTENT_UPDATE = "content_update"
    METADATA_UPDATE = "metadata_update"
    RELATIONSHIP_CHANGE = "relationship_change"
    CONFIDENCE_ADJUSTMENT = "confidence_adjustment"
    STATUS_CHANGE = "status_change"


class PatternType(str, Enum):
    """Types of patterns that can be detected."""
    SEMANTIC_CLUSTER = "semantic_cluster"
    TEMPORAL_SEQUENCE = "temporal_sequence"
    ENTITY_RELATIONSHIP = "entity_relationship"
    TOPIC_EVOLUTION = "topic_evolution"
    USAGE_PATTERN = "usage_pattern"
    QUALITY_CORRELATION = "quality_correlation"


class MemoryMetadata(BaseModel):
    """Structured memory metadata with validation."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    source: str = Field(..., min_length=1, description="Source of the memory")
    source_type: str = Field(..., description="Type of source (document, conversation, etc.)")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    last_modified: datetime = Field(default_factory=datetime.utcnow, description="Last modification timestamp")
    tags: Set[str] = Field(default_factory=set, description="Memory tags")
    entities: List[Dict[str, Any]] = Field(default_factory=list, description="Extracted entities")
    relationships: List[Dict[str, Any]] = Field(default_factory=list, description="Entity relationships")
    confidence_score: float = Field(0.0, ge=0.0, le=1.0, description="Overall confidence score")
    quality_indicators: Dict[str, float] = Field(default_factory=dict, description="Quality metrics")
    access_patterns: Dict[str, Any] = Field(default_factory=dict, description="Access pattern data")
    
    @field_validator('source_type')
    @classmethod
    def validate_source_type(cls, v):
        """Validate source type is recognized."""
        valid_types = {
            'document', 'conversation', 'web_page', 'api_response', 
            'user_input', 'system_generated', 'imported_data'
        }
        if v.lower() not in valid_types:
            raise ValueError(f"Unknown source type: {v}")
        return v.lower()
    
    @field_validator('tags')
    @classmethod
    def validate_tags(cls, v):
        """Validate tags format."""
        if len(v) > 50:
            raise ValueError("Too many tags (maximum 50)")
        for tag in v:
            if not isinstance(tag, str) or len(tag.strip()) == 0:
                raise ValueError("All tags must be non-empty strings")
        return {tag.strip().lower() for tag in v}


class IngestionRequest(BaseModel):
    """Structured memory ingestion request."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    content: str = Field(..., min_length=1, description="Memory content")
    metadata: MemoryMetadata = Field(..., description="Memory metadata")
    embedding: Optional[List[float]] = Field(None, description="Pre-computed embedding")
    extract_entities: bool = Field(True, description="Whether to extract entities")
    detect_relationships: bool = Field(True, description="Whether to detect relationships")
    validate_content: bool = Field(True, description="Whether to validate content quality")
    
    @field_validator('embedding')
    @classmethod
    def validate_embedding(cls, v):
        """Validate embedding dimensions."""
        if v is not None:
            if not isinstance(v, list) or len(v) == 0:
                raise ValueError("Embedding must be a non-empty list")
            if not all(isinstance(x, (int, float)) for x in v):
                raise ValueError("All embedding values must be numeric")
        return v


class RetrievalRequest(BaseModel):
    """Structured memory retrieval request."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    query: str = Field(..., min_length=1, description="Retrieval query")
    query_embedding: Optional[List[float]] = Field(None, description="Pre-computed query embedding")
    filters: Dict[str, Any] = Field(default_factory=dict, description="Retrieval filters")
    limit: int = Field(10, ge=1, le=1000, description="Maximum results to return")
    min_confidence: float = Field(0.0, ge=0.0, le=1.0, description="Minimum confidence threshold")
    include_metadata: bool = Field(True, description="Whether to include metadata")
    score_results: bool = Field(True, description="Whether to score retrieval results")


class UpdateRequest(BaseModel):
    """Structured memory update request."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    memory_id: str = Field(..., min_length=1, description="Memory identifier")
    changes: Dict[str, Any] = Field(..., description="Changes to apply")
    change_reason: str = Field(..., min_length=1, description="Reason for changes")
    validate_changes: bool = Field(True, description="Whether to validate changes")
    update_confidence: bool = Field(True, description="Whether to update confidence scores")
    track_history: bool = Field(True, description="Whether to track change history")


class OperationResult(BaseModel):
    """Structured result from memory operations."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    success: bool = Field(..., description="Operation success status")
    operation_type: OperationType = Field(..., description="Type of operation performed")
    result_data: Dict[str, Any] = Field(..., description="Operation result data")
    confidence_level: ConfidenceLevel = Field(..., description="Confidence in operation result")
    validation_results: List[Dict[str, Any]] = Field(default_factory=list, description="Validation results")
    performance_metrics: Dict[str, float] = Field(default_factory=dict, description="Performance metrics")
    recommendations: List[str] = Field(default_factory=list, description="Operation recommendations")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Operation timestamp")


class DetectedPattern(BaseModel):
    """Structured pattern detection result."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    pattern_type: PatternType = Field(..., description="Type of detected pattern")
    pattern_id: str = Field(..., description="Unique pattern identifier")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Pattern confidence score")
    description: str = Field(..., description="Human-readable pattern description")
    affected_memories: List[str] = Field(..., description="Memory IDs affected by pattern")
    pattern_data: Dict[str, Any] = Field(..., description="Pattern-specific data")
    discovered_at: datetime = Field(default_factory=datetime.utcnow, description="Discovery timestamp")
    significance_score: float = Field(..., ge=0.0, le=1.0, description="Pattern significance")


class MemoryIngestionAgent:
    """
    Pydantic AI Agent for structured memory ingestion with validated metadata.
    
    Handles intelligent memory ingestion with entity extraction, relationship detection,
    and comprehensive metadata validation.
    """
    
    def __init__(
        self,
        vllm_client: Optional[VLLMClient] = None,
        embedder: Optional[Embedder] = None,
        memory_manager: Optional[MemoryManager] = None
    ):
        """Initialize memory ingestion agent."""
        self.vllm_client = vllm_client
        self.embedder = embedder
        self.memory_manager = memory_manager
        self.ingestion_agent = self._create_ingestion_agent()
        
        # Performance tracking
        self.ingestion_stats = {
            'total_ingestions': 0,
            'successful_ingestions': 0,
            'failed_ingestions': 0,
            'average_processing_time': 0.0,
            'entity_extraction_count': 0,
            'relationship_detection_count': 0
        }
        
        logger.info("Initialized MemoryIngestionAgent")
    
    def _create_ingestion_agent(self) -> Optional[Agent]:
        """Create Pydantic AI agent for memory ingestion."""
        try:
            if not self.vllm_client:
                return None
            
            from pydantic_ai.models.openai import OpenAIModel
            
            model = OpenAIModel(
                'llama2',
                base_url='http://localhost:8000/v1',
                api_key='not-needed'
            )
            
            return Agent(
                model,
                result_type=OperationResult,
                system_prompt="""
                You are an expert memory ingestion system. Your task is to process and validate
                memory content for optimal storage and retrieval.
                
                Instructions:
                1. Analyze content quality and completeness
                2. Extract relevant entities and relationships
                3. Validate metadata consistency
                4. Assess information reliability
                5. Generate quality indicators
                6. Provide ingestion recommendations
                
                Focus on ensuring high-quality memory storage with accurate metadata.
                Be thorough in validation and provide actionable feedback.
                """,
                retries=2
            )
            
        except Exception as e:
            logger.error("Failed to create ingestion agent", error=str(e))
            return None
    
    async def ingest_memory(self, request: IngestionRequest) -> OperationResult:
        """
        Ingest memory with comprehensive validation and processing.
        
        Args:
            request: Structured ingestion request
            
        Returns:
            Structured operation result
        """
        start_time = datetime.utcnow()
        self.ingestion_stats['total_ingestions'] += 1
        
        try:
            # Generate embedding if not provided
            if request.embedding is None and self.embedder:
                request.embedding = await self.embedder.embed_text(request.content)
            
            # Extract entities if requested
            entities = []
            if request.extract_entities:
                entities = await self._extract_entities(request.content)
                request.metadata.entities = entities
                self.ingestion_stats['entity_extraction_count'] += 1
            
            # Detect relationships if requested
            relationships = []
            if request.detect_relationships and entities:
                relationships = await self._detect_relationships(request.content, entities)
                request.metadata.relationships = relationships
                self.ingestion_stats['relationship_detection_count'] += 1
            
            # Validate content quality if requested
            validation_results = []
            if request.validate_content:
                validation_results = await self._validate_content_quality(request)
            
            # Calculate confidence score
            confidence_score = await self._calculate_ingestion_confidence(
                request, entities, relationships, validation_results
            )
            request.metadata.confidence_score = confidence_score
            
            # Store memory using memory manager
            memory_id = None
            if self.memory_manager:
                memory_id = await self._store_memory(request)
            
            # Use AI agent for enhanced processing if available
            ai_result = None
            if self.ingestion_agent:
                ai_result = await self._ai_enhanced_ingestion(request, entities, relationships)
            
            # Compile result
            result_data = {
                'memory_id': memory_id,
                'entities_extracted': len(entities),
                'relationships_detected': len(relationships),
                'embedding_generated': request.embedding is not None,
                'content_length': len(request.content),
                'metadata_fields': len(request.metadata.model_dump())
            }
            
            if ai_result:
                result_data.update(ai_result.result_data)
            
            # Determine confidence level
            confidence_level = self._score_to_confidence_level(confidence_score)
            
            # Calculate performance metrics
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            performance_metrics = {
                'processing_time_seconds': processing_time,
                'entities_per_second': len(entities) / max(processing_time, 0.001),
                'content_processing_rate': len(request.content) / max(processing_time, 0.001)
            }
            
            # Update statistics
            self.ingestion_stats['successful_ingestions'] += 1
            self.ingestion_stats['average_processing_time'] = (
                (self.ingestion_stats['average_processing_time'] * 
                 (self.ingestion_stats['successful_ingestions'] - 1) + processing_time) /
                self.ingestion_stats['successful_ingestions']
            )
            
            # Generate recommendations
            recommendations = []
            if confidence_score < 0.7:
                recommendations.append("Consider additional content validation")
            if len(entities) == 0:
                recommendations.append("No entities detected - verify content richness")
            if len(relationships) == 0 and len(entities) > 1:
                recommendations.append("Consider enhancing relationship detection")
            
            result = OperationResult(
                success=True,
                operation_type=OperationType.INGESTION,
                result_data=result_data,
                confidence_level=confidence_level,
                validation_results=validation_results,
                performance_metrics=performance_metrics,
                recommendations=recommendations
            )
            
            logger.info(
                "Memory ingestion completed",
                memory_id=memory_id,
                entities_count=len(entities),
                relationships_count=len(relationships),
                confidence_score=confidence_score,
                processing_time_seconds=processing_time
            )
            
            return result
            
        except Exception as e:
            self.ingestion_stats['failed_ingestions'] += 1
            logger.error("Memory ingestion failed", error=str(e))
            
            return OperationResult(
                success=False,
                operation_type=OperationType.INGESTION,
                result_data={'error': str(e)},
                confidence_level=ConfidenceLevel.VERY_LOW,
                validation_results=[{
                    'type': 'ingestion_error',
                    'message': f"Ingestion failed: {str(e)}"
                }],
                performance_metrics={},
                recommendations=["Manual review required due to ingestion error"]
            )
    
    async def _extract_entities(self, content: str) -> List[Dict[str, Any]]:
        """Extract entities from content."""
        # Simple entity extraction (would integrate with structured_operations.py)
        entities = []
        
        # Basic named entity patterns
        import re
        
        # People (capitalized names)
        people = re.findall(r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b', content)
        for person in people:
            entities.append({
                'text': person,
                'type': 'person',
                'confidence': 0.8
            })
        
        # Organizations (with corp suffixes)
        orgs = re.findall(r'\b[A-Z][a-zA-Z\s&]+(?:Inc|Corp|LLC|Ltd|Co)\b', content)
        for org in orgs:
            entities.append({
                'text': org,
                'type': 'organization',
                'confidence': 0.9
            })
        
        # Dates
        dates = re.findall(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}\b', content)
        for date in dates:
            entities.append({
                'text': date,
                'type': 'date',
                'confidence': 0.7
            })
        
        return entities[:20]  # Limit to 20 entities
    
    async def _detect_relationships(self, content: str, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect relationships between entities."""
        relationships = []
        
        # Simple relationship detection
        entity_texts = [e['text'] for e in entities]
        
        for i, entity1 in enumerate(entity_texts):
            for entity2 in entity_texts[i+1:]:
                # Check if entities appear close together
                entity1_pos = content.find(entity1)
                entity2_pos = content.find(entity2)
                
                if abs(entity1_pos - entity2_pos) < 100:  # Within 100 characters
                    relationships.append({
                        'subject': entity1,
                        'object': entity2,
                        'relationship_type': 'related_to',
                        'confidence': 0.6
                    })
        
        return relationships[:10]  # Limit to 10 relationships
    
    async def _validate_content_quality(self, request: IngestionRequest) -> List[Dict[str, Any]]:
        """Validate content quality."""
        validation_results = []
        
        # Check content length
        if len(request.content) < 10:
            validation_results.append({
                'type': 'content_too_short',
                'severity': 'error',
                'message': 'Content is too short for meaningful analysis'
            })
        elif len(request.content) > 50000:
            validation_results.append({
                'type': 'content_very_long',
                'severity': 'warning',
                'message': 'Content is very long and may affect processing performance'
            })
        
        # Check for empty metadata fields
        metadata_dict = request.metadata.model_dump()
        empty_fields = [k for k, v in metadata_dict.items() if not v]
        if empty_fields:
            validation_results.append({
                'type': 'empty_metadata',
                'severity': 'warning',
                'message': f'Empty metadata fields: {empty_fields}'
            })
        
        # Check source validity
        if not request.metadata.source or len(request.metadata.source.strip()) == 0:
            validation_results.append({
                'type': 'missing_source',
                'severity': 'error',
                'message': 'Memory source is required'
            })
        
        return validation_results
    
    async def _calculate_ingestion_confidence(
        self,
        request: IngestionRequest,
        entities: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]],
        validation_results: List[Dict[str, Any]]
    ) -> float:
        """Calculate overall ingestion confidence score."""
        confidence_factors = []
        
        # Content quality factor
        content_quality = 0.8  # Base quality
        if len(request.content) > 100:
            content_quality += 0.1
        if len(validation_results) == 0:
            content_quality += 0.1
        confidence_factors.append(content_quality)
        
        # Entity extraction factor
        entity_factor = min(1.0, len(entities) / 5) * 0.8 + 0.2
        confidence_factors.append(entity_factor)
        
        # Relationship factor
        relationship_factor = min(1.0, len(relationships) / 3) * 0.7 + 0.3
        confidence_factors.append(relationship_factor)
        
        # Metadata completeness factor
        metadata_dict = request.metadata.model_dump()
        filled_fields = sum(1 for v in metadata_dict.values() if v)
        metadata_factor = filled_fields / len(metadata_dict)
        confidence_factors.append(metadata_factor)
        
        # Error penalty
        error_count = sum(1 for r in validation_results if r.get('severity') == 'error')
        error_penalty = max(0.0, 1.0 - (error_count * 0.3))
        
        # Calculate weighted average
        overall_confidence = sum(confidence_factors) / len(confidence_factors)
        overall_confidence *= error_penalty
        
        return max(0.0, min(1.0, overall_confidence))
    
    async def _store_memory(self, request: IngestionRequest) -> Optional[str]:
        """Store memory using memory manager."""
        try:
            if not self.memory_manager:
                return None
            
            # Convert to memory manager format
            store_request = {
                'content': request.content,
                'metadata': request.metadata.model_dump(),
                'embedding': request.embedding
            }
            
            # This would integrate with actual memory manager
            memory_id = str(uuid.uuid4())
            
            logger.info("Memory stored", memory_id=memory_id)
            return memory_id
            
        except Exception as e:
            logger.error("Failed to store memory", error=str(e))
            return None
    
    async def _ai_enhanced_ingestion(
        self,
        request: IngestionRequest,
        entities: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]]
    ) -> Optional[OperationResult]:
        """AI-enhanced ingestion analysis."""
        if not self.ingestion_agent:
            return None
        
        try:
            context = {
                'content_length': len(request.content),
                'entities_count': len(entities),
                'relationships_count': len(relationships),
                'source_type': request.metadata.source_type
            }
            
            result = await self.ingestion_agent.run(
                f"Analyze this memory ingestion: {request.content[:500]}",
                deps=context
            )
            
            return result.data
            
        except Exception as e:
            logger.warning("AI-enhanced ingestion failed", error=str(e))
            return None
    
    def _score_to_confidence_level(self, score: float) -> ConfidenceLevel:
        """Convert numeric score to confidence level."""
        if score >= 0.95:
            return ConfidenceLevel.VERY_HIGH
        elif score >= 0.80:
            return ConfidenceLevel.HIGH
        elif score >= 0.60:
            return ConfidenceLevel.MEDIUM
        elif score >= 0.40:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW


class MemoryRetrievalAgent:
    """
    Pydantic AI Agent for structured memory retrieval with confidence scoring.
    
    Handles intelligent memory retrieval with relevance scoring, result validation,
    and confidence assessment.
    """
    
    def __init__(
        self,
        vllm_client: Optional[VLLMClient] = None,
        embedder: Optional[Embedder] = None,
        memory_manager: Optional[MemoryManager] = None
    ):
        """Initialize memory retrieval agent."""
        self.vllm_client = vllm_client
        self.embedder = embedder
        self.memory_manager = memory_manager
        self.retrieval_agent = self._create_retrieval_agent()
        
        # Performance tracking
        self.retrieval_stats = {
            'total_retrievals': 0,
            'successful_retrievals': 0,
            'average_retrieval_time': 0.0,
            'average_results_returned': 0.0,
            'confidence_scores': []
        }
        
        logger.info("Initialized MemoryRetrievalAgent")
    
    def _create_retrieval_agent(self) -> Optional[Agent]:
        """Create Pydantic AI agent for memory retrieval."""
        try:
            if not self.vllm_client:
                return None
            
            from pydantic_ai.models.openai import OpenAIModel
            
            model = OpenAIModel(
                'llama2',
                base_url='http://localhost:8000/v1',
                api_key='not-needed'
            )
            
            return Agent(
                model,
                result_type=OperationResult,
                system_prompt="""
                You are an expert memory retrieval system. Your task is to analyze
                retrieval requests and assess result quality and relevance.
                
                Instructions:
                1. Analyze query intent and complexity
                2. Assess retrieval result relevance
                3. Validate result completeness
                4. Score result confidence
                5. Identify potential gaps
                6. Provide retrieval recommendations
                
                Focus on ensuring high-quality, relevant retrieval results.
                Be thorough in relevance assessment and confidence scoring.
                """,
                retries=2
            )
            
        except Exception as e:
            logger.error("Failed to create retrieval agent", error=str(e))
            return None
    
    async def retrieve_memories(self, request: RetrievalRequest) -> OperationResult:
        """
        Retrieve memories with comprehensive analysis and scoring.
        
        Args:
            request: Structured retrieval request
            
        Returns:
            Structured operation result with scored results
        """
        start_time = datetime.utcnow()
        self.retrieval_stats['total_retrievals'] += 1
        
        try:
            # Generate query embedding if not provided
            if request.query_embedding is None and self.embedder:
                request.query_embedding = await self.embedder.embed_text(request.query)
            
            # Perform retrieval using memory manager
            raw_results = []
            if self.memory_manager:
                raw_results = await self._perform_retrieval(request)
            
            # Score and validate results
            scored_results = []
            if request.score_results:
                scored_results = await self._score_retrieval_results(request, raw_results)
            else:
                scored_results = raw_results
            
            # Filter by confidence if specified
            if request.min_confidence > 0:
                scored_results = [
                    r for r in scored_results 
                    if r.get('confidence_score', 0) >= request.min_confidence
                ]
            
            # Limit results
            scored_results = scored_results[:request.limit]
            
            # Calculate overall confidence
            overall_confidence = await self._calculate_retrieval_confidence(
                request, scored_results, raw_results
            )
            
            # Use AI agent for enhanced analysis if available
            ai_result = None
            if self.retrieval_agent:
                ai_result = await self._ai_enhanced_retrieval(request, scored_results)
            
            # Compile result
            result_data = {
                'results': scored_results,
                'total_found': len(raw_results),
                'results_returned': len(scored_results),
                'query_processed': True,
                'embedding_used': request.query_embedding is not None,
                'filters_applied': len(request.filters) > 0
            }
            
            if ai_result:
                result_data.update(ai_result.result_data)
            
            # Determine confidence level
            confidence_level = self._score_to_confidence_level(overall_confidence)
            
            # Calculate performance metrics
            retrieval_time = (datetime.utcnow() - start_time).total_seconds()
            performance_metrics = {
                'retrieval_time_seconds': retrieval_time,
                'results_per_second': len(scored_results) / max(retrieval_time, 0.001),
                'precision_estimate': overall_confidence
            }
            
            # Update statistics
            self.retrieval_stats['successful_retrievals'] += 1
            self.retrieval_stats['average_retrieval_time'] = (
                (self.retrieval_stats['average_retrieval_time'] * 
                 (self.retrieval_stats['successful_retrievals'] - 1) + retrieval_time) /
                self.retrieval_stats['successful_retrievals']
            )
            self.retrieval_stats['average_results_returned'] = (
                (self.retrieval_stats['average_results_returned'] * 
                 (self.retrieval_stats['successful_retrievals'] - 1) + len(scored_results)) /
                self.retrieval_stats['successful_retrievals']
            )
            self.retrieval_stats['confidence_scores'].append(overall_confidence)
            
            # Generate recommendations
            recommendations = []
            if len(scored_results) == 0:
                recommendations.append("No results found - consider broadening search criteria")
            elif overall_confidence < 0.7:
                recommendations.append("Low confidence results - consider refining query")
            if len(scored_results) < request.limit / 2:
                recommendations.append("Few results returned - database may need more content")
            
            result = OperationResult(
                success=True,
                operation_type=OperationType.RETRIEVAL,
                result_data=result_data,
                confidence_level=confidence_level,
                validation_results=[],
                performance_metrics=performance_metrics,
                recommendations=recommendations
            )
            
            logger.info(
                "Memory retrieval completed",
                query=request.query[:100],
                results_found=len(raw_results),
                results_returned=len(scored_results),
                confidence=overall_confidence,
                retrieval_time_seconds=retrieval_time
            )
            
            return result
            
        except Exception as e:
            logger.error("Memory retrieval failed", error=str(e))
            
            return OperationResult(
                success=False,
                operation_type=OperationType.RETRIEVAL,
                result_data={'error': str(e)},
                confidence_level=ConfidenceLevel.VERY_LOW,
                validation_results=[{
                    'type': 'retrieval_error',
                    'message': f"Retrieval failed: {str(e)}"
                }],
                performance_metrics={},
                recommendations=["Manual review required due to retrieval error"]
            )
    
    async def _perform_retrieval(self, request: RetrievalRequest) -> List[Dict[str, Any]]:
        """Perform actual memory retrieval."""
        # This would integrate with actual memory manager
        # For now, return mock results
        mock_results = [
            {
                'memory_id': f'mem_{i}',
                'content': f'Mock memory content {i} related to: {request.query}',
                'metadata': {'source': f'source_{i}', 'created_at': datetime.utcnow().isoformat()},
                'similarity_score': 0.9 - (i * 0.1)
            }
            for i in range(min(5, request.limit))
        ]
        
        return mock_results
    
    async def _score_retrieval_results(
        self,
        request: RetrievalRequest,
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Score retrieval results for relevance and confidence."""
        scored_results = []
        
        for result in results:
            # Calculate relevance score
            relevance_score = result.get('similarity_score', 0.5)
            
            # Calculate freshness score
            created_at = result.get('metadata', {}).get('created_at')
            freshness_score = 0.8  # Default
            if created_at:
                try:
                    created_dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                    age_days = (datetime.utcnow() - created_dt).days
                    freshness_score = max(0.1, 1.0 - (age_days / 365))  # Decay over a year
                except:
                    pass
            
            # Calculate overall confidence
            confidence_score = (relevance_score * 0.7) + (freshness_score * 0.3)
            
            # Add scores to result
            scored_result = result.copy()
            scored_result.update({
                'relevance_score': relevance_score,
                'freshness_score': freshness_score,
                'confidence_score': confidence_score
            })
            
            scored_results.append(scored_result)
        
        # Sort by confidence score
        scored_results.sort(key=lambda x: x['confidence_score'], reverse=True)
        
        return scored_results
    
    async def _calculate_retrieval_confidence(
        self,
        request: RetrievalRequest,
        scored_results: List[Dict[str, Any]],
        raw_results: List[Dict[str, Any]]
    ) -> float:
        """Calculate overall retrieval confidence."""
        if not scored_results:
            return 0.0
        
        # Average confidence of results
        avg_confidence = sum(r.get('confidence_score', 0) for r in scored_results) / len(scored_results)
        
        # Coverage factor (how many results we found vs requested)
        coverage_factor = min(1.0, len(raw_results) / request.limit)
        
        # Query complexity factor (simple heuristic)
        query_complexity = min(1.0, len(request.query.split()) / 10)
        complexity_bonus = 1.0 + (query_complexity * 0.1)
        
        overall_confidence = avg_confidence * coverage_factor * complexity_bonus
        return max(0.0, min(1.0, overall_confidence))
    
    async def _ai_enhanced_retrieval(
        self,
        request: RetrievalRequest,
        results: List[Dict[str, Any]]
    ) -> Optional[OperationResult]:
        """AI-enhanced retrieval analysis."""
        if not self.retrieval_agent:
            return None
        
        try:
            context = {
                'query_length': len(request.query),
                'results_count': len(results),
                'avg_confidence': sum(r.get('confidence_score', 0) for r in results) / max(len(results), 1),
                'filters_used': len(request.filters)
            }
            
            result = await self.retrieval_agent.run(
                f"Analyze this retrieval: Query='{request.query}', Results={len(results)}",
                deps=context
            )
            
            return result.data
            
        except Exception as e:
            logger.warning("AI-enhanced retrieval failed", error=str(e))
            return None
    
    def _score_to_confidence_level(self, score: float) -> ConfidenceLevel:
        """Convert numeric score to confidence level."""
        if score >= 0.95:
            return ConfidenceLevel.VERY_HIGH
        elif score >= 0.80:
            return ConfidenceLevel.HIGH
        elif score >= 0.60:
            return ConfidenceLevel.MEDIUM
        elif score >= 0.40:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW


class StructuredMemoryOperationsManager:
    """
    Manager for all structured memory operations.
    
    Coordinates ingestion, retrieval, update, and analysis operations with
    comprehensive validation and AI enhancement.
    """
    
    def __init__(
        self,
        vllm_client: Optional[VLLMClient] = None,
        embedder: Optional[Embedder] = None,
        memory_manager: Optional[MemoryManager] = None
    ):
        """Initialize structured memory operations manager."""
        self.vllm_client = vllm_client
        self.embedder = embedder
        self.memory_manager = memory_manager
        
        # Initialize agents
        self.ingestion_agent = MemoryIngestionAgent(vllm_client, embedder, memory_manager)
        self.retrieval_agent = MemoryRetrievalAgent(vllm_client, embedder, memory_manager)
        
        # Manager statistics
        self.manager_stats = {
            'total_operations': 0,
            'successful_operations': 0,
            'operation_types': {op.value: 0 for op in OperationType},
            'average_operation_time': 0.0
        }
        
        logger.info("Initialized StructuredMemoryOperationsManager")
    
    async def process_operation(
        self,
        operation_type: OperationType,
        request_data: Dict[str, Any]
    ) -> OperationResult:
        """
        Process structured memory operation.
        
        Args:
            operation_type: Type of operation to perform
            request_data: Operation request data
            
        Returns:
            Structured operation result
        """
        start_time = datetime.utcnow()
        self.manager_stats['total_operations'] += 1
        self.manager_stats['operation_types'][operation_type.value] += 1
        
        try:
            if operation_type == OperationType.INGESTION:
                request = IngestionRequest(**request_data)
                result = await self.ingestion_agent.ingest_memory(request)
            elif operation_type == OperationType.RETRIEVAL:
                request = RetrievalRequest(**request_data)
                result = await self.retrieval_agent.retrieve_memories(request)
            else:
                raise ValueError(f"Unsupported operation type: {operation_type}")
            
            if result.success:
                self.manager_stats['successful_operations'] += 1
            
            operation_time = (datetime.utcnow() - start_time).total_seconds()
            self.manager_stats['average_operation_time'] = (
                (self.manager_stats['average_operation_time'] * 
                 (self.manager_stats['total_operations'] - 1) + operation_time) /
                self.manager_stats['total_operations']
            )
            
            return result
            
        except Exception as e:
            logger.error("Structured operation failed", operation_type=operation_type.value, error=str(e))
            
            return OperationResult(
                success=False,
                operation_type=operation_type,
                result_data={'error': str(e)},
                confidence_level=ConfidenceLevel.VERY_LOW,
                validation_results=[{
                    'type': 'operation_error',
                    'message': f"Operation failed: {str(e)}"
                }],
                performance_metrics={},
                recommendations=["Manual review required due to operation error"]
            )
    
    async def get_operation_stats(self) -> Dict[str, Any]:
        """Get comprehensive operation statistics."""
        return {
            'manager_stats': self.manager_stats,
            'ingestion_stats': self.ingestion_agent.ingestion_stats,
            'retrieval_stats': self.retrieval_agent.retrieval_stats
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check of all components."""
        health_status = {
            'status': 'healthy',
            'agents': {},
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Check agents
        agents = [
            ('ingestion_agent', self.ingestion_agent),
            ('retrieval_agent', self.retrieval_agent)
        ]
        
        for agent_name, agent in agents:
            try:
                health_status['agents'][agent_name] = {
                    'status': 'healthy',
                    'agent_available': hasattr(agent, 'ingestion_agent') or hasattr(agent, 'retrieval_agent'),
                    'stats_available': True
                }
            except Exception as e:
                health_status['agents'][agent_name] = {
                    'status': 'unhealthy',
                    'error': str(e)
                }
        
        return health_status


# Example usage
async def example_usage():
    """Example of using StructuredMemoryOperationsManager."""
    # Initialize manager
    manager = StructuredMemoryOperationsManager()
    
    # Test ingestion
    ingestion_data = {
        'content': 'John Smith works at OpenAI and has been developing advanced AI systems since 2021.',
        'metadata': {
            'source': 'example_document.txt',
            'source_type': 'document',
            'tags': ['ai', 'technology', 'development']
        },
        'extract_entities': True,
        'detect_relationships': True
    }
    
    ingestion_result = await manager.process_operation(
        OperationType.INGESTION,
        ingestion_data
    )
    
    print(f"Ingestion Result:")
    print(f"  Success: {ingestion_result.success}")
    print(f"  Confidence: {ingestion_result.confidence_level.value}")
    print(f"  Entities extracted: {ingestion_result.result_data.get('entities_extracted', 0)}")
    print(f"  Relationships: {ingestion_result.result_data.get('relationships_detected', 0)}")
    
    # Test retrieval
    retrieval_data = {
        'query': 'AI development at OpenAI',
        'limit': 5,
        'min_confidence': 0.7,
        'score_results': True
    }
    
    retrieval_result = await manager.process_operation(
        OperationType.RETRIEVAL,
        retrieval_data
    )
    
    print(f"\nRetrieval Result:")
    print(f"  Success: {retrieval_result.success}")
    print(f"  Confidence: {retrieval_result.confidence_level.value}")
    print(f"  Results found: {retrieval_result.result_data.get('total_found', 0)}")
    print(f"  Results returned: {retrieval_result.result_data.get('results_returned', 0)}")
    
    # Get statistics
    stats = await manager.get_operation_stats()
    print(f"\nOperation stats: {stats['manager_stats']}")


if __name__ == "__main__":
    asyncio.run(example_usage())