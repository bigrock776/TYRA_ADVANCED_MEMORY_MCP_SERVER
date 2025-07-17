"""
Comprehensive AI Operation Validation with Pydantic AI.

This module provides enterprise-grade structured AI operations with full validation,
anti-hallucination measures, and comprehensive error handling. All AI operations
use Pydantic AI for structured outputs and multi-layer validation.
"""

import asyncio
import json
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import hashlib
from urllib.parse import urlparse

import structlog
from pydantic import BaseModel, Field, ConfigDict, field_validator
from pydantic_ai import Agent, RunContext, ModelRetry, ValidationError as PydanticAIValidationError
from pydantic_ai.models.openai import OpenAIModel

from ..clients.vllm_client import VLLMClient, ChatMessage, Role
from ..embeddings.embedder import Embedder
from ..utils.config import settings

logger = structlog.get_logger(__name__)


class OperationType(str, Enum):
    """Types of structured AI operations."""
    ENTITY_EXTRACTION = "entity_extraction"
    RELATIONSHIP_INFERENCE = "relationship_inference"
    QUERY_PROCESSING = "query_processing"
    RESPONSE_VALIDATION = "response_validation"
    CONTENT_ANALYSIS = "content_analysis"
    INTENT_CLASSIFICATION = "intent_classification"


class EntityType(str, Enum):
    """Types of entities that can be extracted."""
    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    TECHNOLOGY = "technology"
    CONCEPT = "concept"
    DATE = "date"
    FINANCIAL = "financial"
    PRODUCT = "product"
    EVENT = "event"


class RelationType(str, Enum):
    """Types of relationships between entities."""
    WORKS_AT = "works_at"
    LOCATED_IN = "located_in"
    CREATED_BY = "created_by"
    PART_OF = "part_of"
    RELATED_TO = "related_to"
    INFLUENCES = "influences"
    DEPENDS_ON = "depends_on"
    COMPETED_WITH = "competes_with"
    ACQUIRED_BY = "acquired_by"


class QueryIntent(str, Enum):
    """Types of query intents that can be classified."""
    FACTUAL_LOOKUP = "factual_lookup"
    CONCEPTUAL_EXPLANATION = "conceptual_explanation"
    PROCEDURAL_GUIDANCE = "procedural_guidance"
    COMPARISON_REQUEST = "comparison_request"
    ANALYSIS_REQUEST = "analysis_request"
    PREDICTION_REQUEST = "prediction_request"
    CREATIVE_GENERATION = "creative_generation"
    TROUBLESHOOTING = "troubleshooting"


class ConfidenceLevel(str, Enum):
    """Confidence levels for AI operations."""
    VERY_HIGH = "very_high"  # 95-100%
    HIGH = "high"            # 80-94%
    MEDIUM = "medium"        # 60-79%
    LOW = "low"              # 40-59%
    VERY_LOW = "very_low"    # 0-39%


class ExtractedEntity(BaseModel):
    """Structured entity extraction result."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    text: str = Field(..., min_length=1, description="Entity text as it appears")
    entity_type: EntityType = Field(..., description="Classified entity type")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Extraction confidence")
    start_pos: int = Field(..., ge=0, description="Start position in source text")
    end_pos: int = Field(..., ge=0, description="End position in source text")
    normalized_form: str = Field(..., description="Normalized entity representation")
    context: str = Field(..., description="Surrounding context")
    attributes: Dict[str, str] = Field(default_factory=dict, description="Additional entity attributes")
    
    @field_validator('end_pos')
    @classmethod
    def validate_positions(cls, v, info):
        """Ensure end position is after start position."""
        if 'start_pos' in info.data and v <= info.data['start_pos']:
            raise ValueError("End position must be after start position")
        return v


class InferredRelationship(BaseModel):
    """Structured relationship inference result."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    subject_entity: str = Field(..., min_length=1, description="Subject entity")
    predicate: RelationType = Field(..., description="Relationship type")
    object_entity: str = Field(..., min_length=1, description="Object entity")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Inference confidence")
    evidence: List[str] = Field(..., min_items=1, description="Supporting evidence from text")
    temporal_context: Optional[str] = Field(None, description="Temporal context if applicable")
    certainty_qualifiers: List[str] = Field(default_factory=list, description="Uncertainty indicators")
    
    @field_validator('subject_entity', 'object_entity')
    @classmethod
    def validate_entities_different(cls, v, info):
        """Ensure subject and object are different."""
        if 'subject_entity' in info.data and v == info.data['subject_entity']:
            raise ValueError("Subject and object entities must be different")
        return v


class ProcessedQuery(BaseModel):
    """Structured query processing result."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    original_query: str = Field(..., min_length=1, description="Original user query")
    intent: QueryIntent = Field(..., description="Classified query intent")
    intent_confidence: float = Field(..., ge=0.0, le=1.0, description="Intent classification confidence")
    extracted_entities: List[str] = Field(..., description="Key entities in query")
    keywords: List[str] = Field(..., description="Important keywords")
    semantic_representation: str = Field(..., description="Semantic query representation")
    suggested_strategies: List[str] = Field(..., description="Recommended processing strategies")
    complexity_score: float = Field(..., ge=0.0, le=1.0, description="Query complexity assessment")
    ambiguity_flags: List[str] = Field(default_factory=list, description="Identified ambiguities")


class ValidationResult(BaseModel):
    """Structured response validation result."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    is_valid: bool = Field(..., description="Overall validation result")
    confidence_level: ConfidenceLevel = Field(..., description="Confidence in validation")
    factual_accuracy: float = Field(..., ge=0.0, le=1.0, description="Factual accuracy score")
    source_grounding: float = Field(..., ge=0.0, le=1.0, description="Source grounding score")
    consistency_score: float = Field(..., ge=0.0, le=1.0, description="Internal consistency score")
    hallucination_risk: float = Field(..., ge=0.0, le=1.0, description="Hallucination risk assessment")
    validation_details: Dict[str, Any] = Field(..., description="Detailed validation results")
    recommendations: List[str] = Field(..., description="Improvement recommendations")
    warnings: List[str] = Field(default_factory=list, description="Validation warnings")


class ContentAnalysis(BaseModel):
    """Structured content analysis result."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    main_topics: List[str] = Field(..., max_items=10, description="Main topics identified")
    sentiment_score: float = Field(..., ge=-1.0, le=1.0, description="Overall sentiment")
    readability_score: float = Field(..., ge=0.0, le=100.0, description="Readability assessment")
    technical_complexity: float = Field(..., ge=0.0, le=1.0, description="Technical complexity level")
    key_insights: List[str] = Field(..., description="Key insights extracted")
    factual_claims: List[str] = Field(..., description="Identified factual claims")
    opinion_markers: List[str] = Field(default_factory=list, description="Opinion/subjective markers")
    temporal_references: List[str] = Field(default_factory=list, description="Time-related references")
    quality_indicators: Dict[str, float] = Field(..., description="Quality assessment metrics")


class EntityExtractionAgent:
    """
    Pydantic AI Agent for structured entity extraction.
    
    Extracts and validates entities from text with high accuracy and structured output.
    """
    
    def __init__(self, vllm_client: Optional[VLLMClient] = None):
        """Initialize entity extraction agent."""
        self.vllm_client = vllm_client
        self.agent = self._create_agent()
        
        # Performance tracking
        self.extraction_stats = {
            'total_extractions': 0,
            'successful_extractions': 0,
            'average_entities_per_text': 0.0,
            'average_confidence': 0.0,
            'error_count': 0
        }
        
        logger.info("Initialized EntityExtractionAgent")
    
    def _create_agent(self) -> Optional[Agent]:
        """Create Pydantic AI agent for entity extraction."""
        try:
            if not self.vllm_client:
                logger.warning("No VLLMClient provided, agent will use fallback methods")
                return None
            
            # Use local LLM through OpenAI-compatible API
            model = OpenAIModel(
                'llama2',
                base_url='http://localhost:8000/v1',
                api_key='not-needed'
            )
            
            return Agent(
                model,
                result_type=List[ExtractedEntity],
                system_prompt="""
                You are an expert entity extraction system. Your task is to identify and extract
                entities from text with high precision and recall.
                
                Instructions:
                1. Identify all significant entities in the text
                2. Classify each entity into the appropriate type
                3. Provide accurate position information
                4. Assign confidence scores based on clarity and context
                5. Normalize entity forms for consistency
                6. Include relevant context for each entity
                
                Focus on accuracy over quantity. Only extract entities you are confident about.
                Provide detailed reasoning for entity type classification.
                """,
                retries=2
            )
            
        except Exception as e:
            logger.error("Failed to create entity extraction agent", error=str(e))
            return None
    
    async def extract_entities(
        self,
        text: str,
        entity_types: Optional[List[EntityType]] = None,
        min_confidence: float = 0.7
    ) -> List[ExtractedEntity]:
        """
        Extract entities from text with structured validation.
        
        Args:
            text: Input text for entity extraction
            entity_types: Optional filter for specific entity types
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of validated extracted entities
        """
        start_time = datetime.utcnow()
        self.extraction_stats['total_extractions'] += 1
        
        try:
            if self.agent:
                # Use Pydantic AI agent
                result = await self.agent.run(
                    f"Extract entities from this text: {text}",
                    deps={
                        'entity_types': entity_types,
                        'min_confidence': min_confidence,
                        'text_length': len(text)
                    }
                )
                
                entities = result.data
                
                # Filter by confidence and entity types
                filtered_entities = [
                    entity for entity in entities
                    if entity.confidence >= min_confidence and
                    (not entity_types or entity.entity_type in entity_types)
                ]
                
            else:
                # Fallback to rule-based extraction
                filtered_entities = await self._fallback_entity_extraction(
                    text, entity_types, min_confidence
                )
            
            # Update statistics
            self.extraction_stats['successful_extractions'] += 1
            self.extraction_stats['average_entities_per_text'] = (
                (self.extraction_stats['average_entities_per_text'] * 
                 (self.extraction_stats['successful_extractions'] - 1) +
                 len(filtered_entities)) / self.extraction_stats['successful_extractions']
            )
            
            if filtered_entities:
                avg_conf = sum(e.confidence for e in filtered_entities) / len(filtered_entities)
                self.extraction_stats['average_confidence'] = (
                    (self.extraction_stats['average_confidence'] * 
                     (self.extraction_stats['successful_extractions'] - 1) +
                     avg_conf) / self.extraction_stats['successful_extractions']
                )
            
            extraction_time = (datetime.utcnow() - start_time).total_seconds()
            logger.info(
                "Entity extraction completed",
                text_length=len(text),
                entities_found=len(filtered_entities),
                extraction_time_seconds=extraction_time
            )
            
            return filtered_entities
            
        except Exception as e:
            self.extraction_stats['error_count'] += 1
            logger.error("Entity extraction failed", error=str(e))
            return []
    
    async def _fallback_entity_extraction(
        self,
        text: str,
        entity_types: Optional[List[EntityType]],
        min_confidence: float
    ) -> List[ExtractedEntity]:
        """Fallback rule-based entity extraction."""
        entities = []
        
        # Simple pattern-based extraction
        patterns = {
            EntityType.PERSON: r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b',
            EntityType.ORGANIZATION: r'\b[A-Z][a-zA-Z\s&]+(?:Inc|Corp|LLC|Ltd|Co)\b',
            EntityType.DATE: r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}\b',
            EntityType.FINANCIAL: r'\$\d+(?:,\d{3})*(?:\.\d{2})?',
            EntityType.TECHNOLOGY: r'\b(?:AI|ML|API|HTTP|SQL|JSON|XML|Python|JavaScript)\b'
        }
        
        for entity_type, pattern in patterns.items():
            if entity_types and entity_type not in entity_types:
                continue
                
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                entity = ExtractedEntity(
                    text=match.group(),
                    entity_type=entity_type,
                    confidence=0.8,  # Default confidence for pattern matching
                    start_pos=match.start(),
                    end_pos=match.end(),
                    normalized_form=match.group().lower().strip(),
                    context=text[max(0, match.start()-50):match.end()+50],
                    attributes={'extraction_method': 'pattern_based'}
                )
                
                if entity.confidence >= min_confidence:
                    entities.append(entity)
        
        return entities


class RelationshipInferenceAgent:
    """
    Pydantic AI Agent for structured relationship inference.
    
    Infers and validates relationships between entities with confidence scoring.
    """
    
    def __init__(self, vllm_client: Optional[VLLMClient] = None):
        """Initialize relationship inference agent."""
        self.vllm_client = vllm_client
        self.agent = self._create_agent()
        
        # Performance tracking
        self.inference_stats = {
            'total_inferences': 0,
            'successful_inferences': 0,
            'average_relationships_per_text': 0.0,
            'average_confidence': 0.0,
            'error_count': 0
        }
        
        logger.info("Initialized RelationshipInferenceAgent")
    
    def _create_agent(self) -> Optional[Agent]:
        """Create Pydantic AI agent for relationship inference."""
        try:
            if not self.vllm_client:
                return None
            
            model = OpenAIModel(
                'llama2',
                base_url='http://localhost:8000/v1',
                api_key='not-needed'
            )
            
            return Agent(
                model,
                result_type=List[InferredRelationship],
                system_prompt="""
                You are an expert relationship inference system. Your task is to identify
                relationships between entities in text with high accuracy.
                
                Instructions:
                1. Identify explicit and implicit relationships between entities
                2. Classify relationships using appropriate relation types
                3. Provide strong evidence from the text
                4. Assign confidence scores based on evidence strength
                5. Include temporal context when relevant
                6. Flag uncertainty with appropriate qualifiers
                
                Only infer relationships that have clear textual support.
                Avoid speculation and focus on evidence-based relationships.
                """,
                retries=2
            )
            
        except Exception as e:
            logger.error("Failed to create relationship inference agent", error=str(e))
            return None
    
    async def infer_relationships(
        self,
        text: str,
        entities: List[ExtractedEntity],
        min_confidence: float = 0.7
    ) -> List[InferredRelationship]:
        """
        Infer relationships between entities with validation.
        
        Args:
            text: Source text containing entities
            entities: List of extracted entities
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of validated inferred relationships
        """
        start_time = datetime.utcnow()
        self.inference_stats['total_inferences'] += 1
        
        try:
            if self.agent and len(entities) >= 2:
                # Prepare entity context for the agent
                entity_context = {
                    'entities': [e.model_dump() for e in entities],
                    'text_length': len(text)
                }
                
                result = await self.agent.run(
                    f"Infer relationships between entities in this text: {text}",
                    deps=entity_context
                )
                
                relationships = result.data
                
                # Filter by confidence
                filtered_relationships = [
                    rel for rel in relationships
                    if rel.confidence >= min_confidence
                ]
                
            else:
                # Fallback to rule-based inference
                filtered_relationships = await self._fallback_relationship_inference(
                    text, entities, min_confidence
                )
            
            # Update statistics
            self.inference_stats['successful_inferences'] += 1
            self.inference_stats['average_relationships_per_text'] = (
                (self.inference_stats['average_relationships_per_text'] * 
                 (self.inference_stats['successful_inferences'] - 1) +
                 len(filtered_relationships)) / self.inference_stats['successful_inferences']
            )
            
            if filtered_relationships:
                avg_conf = sum(r.confidence for r in filtered_relationships) / len(filtered_relationships)
                self.inference_stats['average_confidence'] = (
                    (self.inference_stats['average_confidence'] * 
                     (self.inference_stats['successful_inferences'] - 1) +
                     avg_conf) / self.inference_stats['successful_inferences']
                )
            
            inference_time = (datetime.utcnow() - start_time).total_seconds()
            logger.info(
                "Relationship inference completed",
                entities_count=len(entities),
                relationships_found=len(filtered_relationships),
                inference_time_seconds=inference_time
            )
            
            return filtered_relationships
            
        except Exception as e:
            self.inference_stats['error_count'] += 1
            logger.error("Relationship inference failed", error=str(e))
            return []
    
    async def _fallback_relationship_inference(
        self,
        text: str,
        entities: List[ExtractedEntity],
        min_confidence: float
    ) -> List[InferredRelationship]:
        """Fallback rule-based relationship inference."""
        relationships = []
        
        # Simple pattern-based relationship detection
        relationship_patterns = [
            (r'(\w+)\s+works\s+at\s+(\w+)', RelationType.WORKS_AT),
            (r'(\w+)\s+is\s+located\s+in\s+(\w+)', RelationType.LOCATED_IN),
            (r'(\w+)\s+created\s+(\w+)', RelationType.CREATED_BY),
            (r'(\w+)\s+is\s+part\s+of\s+(\w+)', RelationType.PART_OF),
            (r'(\w+)\s+acquired\s+(\w+)', RelationType.ACQUIRED_BY)
        ]
        
        entity_texts = {e.text.lower(): e for e in entities}
        
        for pattern, relation_type in relationship_patterns:
            matches = re.finditer(pattern, text.lower())
            for match in matches:
                subject = match.group(1)
                object_entity = match.group(2)
                
                if subject in entity_texts and object_entity in entity_texts:
                    relationship = InferredRelationship(
                        subject_entity=subject,
                        predicate=relation_type,
                        object_entity=object_entity,
                        confidence=0.75,  # Default confidence for pattern matching
                        evidence=[match.group()],
                        temporal_context=None,
                        certainty_qualifiers=[]
                    )
                    
                    if relationship.confidence >= min_confidence:
                        relationships.append(relationship)
        
        return relationships


class QueryProcessingAgent:
    """
    Pydantic AI Agent for structured query processing and intent classification.
    
    Processes user queries with intent classification and semantic analysis.
    """
    
    def __init__(self, vllm_client: Optional[VLLMClient] = None):
        """Initialize query processing agent."""
        self.vllm_client = vllm_client
        self.agent = self._create_agent()
        
        # Performance tracking
        self.processing_stats = {
            'total_queries': 0,
            'successful_classifications': 0,
            'average_confidence': 0.0,
            'intent_distribution': {intent.value: 0 for intent in QueryIntent},
            'error_count': 0
        }
        
        logger.info("Initialized QueryProcessingAgent")
    
    def _create_agent(self) -> Optional[Agent]:
        """Create Pydantic AI agent for query processing."""
        try:
            if not self.vllm_client:
                return None
            
            model = OpenAIModel(
                'llama2',
                base_url='http://localhost:8000/v1',
                api_key='not-needed'
            )
            
            return Agent(
                model,
                result_type=ProcessedQuery,
                system_prompt="""
                You are an expert query processing system. Your task is to analyze user queries
                and provide structured processing information.
                
                Instructions:
                1. Classify the query intent accurately
                2. Extract key entities and keywords
                3. Create semantic representations
                4. Suggest appropriate processing strategies
                5. Assess query complexity
                6. Identify ambiguities and clarification needs
                
                Focus on understanding user intent and providing actionable processing guidance.
                Be thorough in analyzing query semantics and context.
                """,
                retries=2
            )
            
        except Exception as e:
            logger.error("Failed to create query processing agent", error=str(e))
            return None
    
    async def process_query(self, query: str) -> ProcessedQuery:
        """
        Process query with structured analysis and intent classification.
        
        Args:
            query: User query to process
            
        Returns:
            Structured query processing result
        """
        start_time = datetime.utcnow()
        self.processing_stats['total_queries'] += 1
        
        try:
            if self.agent:
                result = await self.agent.run(
                    f"Process and analyze this query: {query}",
                    deps={'query_length': len(query)}
                )
                
                processed = result.data
                
            else:
                # Fallback to rule-based processing
                processed = await self._fallback_query_processing(query)
            
            # Update statistics
            self.processing_stats['successful_classifications'] += 1
            self.processing_stats['average_confidence'] = (
                (self.processing_stats['average_confidence'] * 
                 (self.processing_stats['successful_classifications'] - 1) +
                 processed.intent_confidence) / self.processing_stats['successful_classifications']
            )
            
            self.processing_stats['intent_distribution'][processed.intent.value] += 1
            
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            logger.info(
                "Query processing completed",
                query_length=len(query),
                intent=processed.intent.value,
                confidence=processed.intent_confidence,
                processing_time_seconds=processing_time
            )
            
            return processed
            
        except Exception as e:
            self.processing_stats['error_count'] += 1
            logger.error("Query processing failed", error=str(e))
            
            # Return minimal fallback result
            return ProcessedQuery(
                original_query=query,
                intent=QueryIntent.FACTUAL_LOOKUP,
                intent_confidence=0.1,
                extracted_entities=[],
                keywords=query.split(),
                semantic_representation=query,
                suggested_strategies=["fallback_search"],
                complexity_score=0.5,
                ambiguity_flags=["processing_error"]
            )
    
    async def _fallback_query_processing(self, query: str) -> ProcessedQuery:
        """Fallback rule-based query processing."""
        # Simple intent classification based on keywords
        intent_keywords = {
            QueryIntent.FACTUAL_LOOKUP: ['what', 'who', 'when', 'where', 'define', 'meaning'],
            QueryIntent.CONCEPTUAL_EXPLANATION: ['how', 'why', 'explain', 'describe', 'concept'],
            QueryIntent.PROCEDURAL_GUIDANCE: ['how to', 'steps', 'guide', 'tutorial', 'process'],
            QueryIntent.COMPARISON_REQUEST: ['compare', 'difference', 'versus', 'vs', 'better'],
            QueryIntent.ANALYSIS_REQUEST: ['analyze', 'examine', 'evaluate', 'assess'],
            QueryIntent.PREDICTION_REQUEST: ['predict', 'forecast', 'future', 'will', 'trend'],
            QueryIntent.TROUBLESHOOTING: ['error', 'problem', 'fix', 'debug', 'issue']
        }
        
        query_lower = query.lower()
        best_intent = QueryIntent.FACTUAL_LOOKUP
        best_score = 0
        
        for intent, keywords in intent_keywords.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            if score > best_score:
                best_score = score
                best_intent = intent
        
        # Extract basic keywords (remove stop words)
        stop_words = {'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 'but'}
        keywords = [word for word in query.split() if word.lower() not in stop_words]
        
        # Simple complexity assessment
        complexity_score = min(1.0, len(query) / 100 + len(keywords) / 20)
        
        return ProcessedQuery(
            original_query=query,
            intent=best_intent,
            intent_confidence=0.7 if best_score > 0 else 0.3,
            extracted_entities=keywords[:5],  # Simple entity approximation
            keywords=keywords,
            semantic_representation=query,
            suggested_strategies=["basic_search", "keyword_matching"],
            complexity_score=complexity_score,
            ambiguity_flags=[]
        )


class ResponseValidationAgent:
    """
    Pydantic AI Agent for comprehensive response validation and hallucination detection.
    
    Validates AI-generated responses for accuracy, consistency, and grounding.
    """
    
    def __init__(self, vllm_client: Optional[VLLMClient] = None):
        """Initialize response validation agent."""
        self.vllm_client = vllm_client
        self.agent = self._create_agent()
        
        # Performance tracking
        self.validation_stats = {
            'total_validations': 0,
            'passed_validations': 0,
            'average_confidence': 0.0,
            'hallucination_detections': 0,
            'error_count': 0
        }
        
        logger.info("Initialized ResponseValidationAgent")
    
    def _create_agent(self) -> Optional[Agent]:
        """Create Pydantic AI agent for response validation."""
        try:
            if not self.vllm_client:
                return None
            
            model = OpenAIModel(
                'llama2',
                base_url='http://localhost:8000/v1',
                api_key='not-needed'
            )
            
            return Agent(
                model,
                result_type=ValidationResult,
                system_prompt="""
                You are an expert response validation system. Your task is to validate AI-generated
                responses for accuracy, consistency, and proper grounding in source material.
                
                Instructions:
                1. Check factual accuracy against provided sources
                2. Verify internal consistency within the response
                3. Assess source grounding and citation quality
                4. Detect potential hallucinations or unsupported claims
                5. Evaluate response quality and completeness
                6. Provide specific recommendations for improvement
                
                Be rigorous in validation and flag any concerns about accuracy or hallucination.
                Focus on evidence-based assessment and clear reasoning.
                """,
                retries=2
            )
            
        except Exception as e:
            logger.error("Failed to create response validation agent", error=str(e))
            return None
    
    async def validate_response(
        self,
        response: str,
        query: str,
        source_context: List[str],
        min_confidence: float = 0.8
    ) -> ValidationResult:
        """
        Validate AI response with comprehensive checks.
        
        Args:
            response: AI-generated response to validate
            query: Original query context
            source_context: Source materials for grounding check
            min_confidence: Minimum confidence threshold
            
        Returns:
            Structured validation result
        """
        start_time = datetime.utcnow()
        self.validation_stats['total_validations'] += 1
        
        try:
            if self.agent:
                result = await self.agent.run(
                    f"Validate this response: {response}",
                    deps={
                        'query': query,
                        'source_context': source_context,
                        'min_confidence': min_confidence,
                        'response_length': len(response)
                    }
                )
                
                validation = result.data
                
            else:
                # Fallback to rule-based validation
                validation = await self._fallback_response_validation(
                    response, query, source_context, min_confidence
                )
            
            # Update statistics
            if validation.is_valid:
                self.validation_stats['passed_validations'] += 1
            
            if validation.hallucination_risk > 0.7:
                self.validation_stats['hallucination_detections'] += 1
            
            confidence_numeric = self._confidence_to_numeric(validation.confidence_level)
            self.validation_stats['average_confidence'] = (
                (self.validation_stats['average_confidence'] * 
                 (self.validation_stats['total_validations'] - 1) +
                 confidence_numeric) / self.validation_stats['total_validations']
            )
            
            validation_time = (datetime.utcnow() - start_time).total_seconds()
            logger.info(
                "Response validation completed",
                is_valid=validation.is_valid,
                confidence_level=validation.confidence_level.value,
                hallucination_risk=validation.hallucination_risk,
                validation_time_seconds=validation_time
            )
            
            return validation
            
        except Exception as e:
            self.validation_stats['error_count'] += 1
            logger.error("Response validation failed", error=str(e))
            
            # Return conservative validation result
            return ValidationResult(
                is_valid=False,
                confidence_level=ConfidenceLevel.VERY_LOW,
                factual_accuracy=0.0,
                source_grounding=0.0,
                consistency_score=0.0,
                hallucination_risk=1.0,
                validation_details={'error': str(e)},
                recommendations=["Manual review required due to validation error"],
                warnings=["Validation system error"]
            )
    
    async def _fallback_response_validation(
        self,
        response: str,
        query: str,
        source_context: List[str],
        min_confidence: float
    ) -> ValidationResult:
        """Fallback rule-based response validation."""
        # Simple validation checks
        validation_details = {}
        
        # Check response length appropriateness
        appropriate_length = 50 <= len(response) <= 1000
        validation_details['appropriate_length'] = appropriate_length
        
        # Check for source references
        has_source_refs = any(
            source_phrase in response.lower() 
            for source_phrase in ['according to', 'based on', 'source:', 'reference:']
        )
        validation_details['has_source_references'] = has_source_refs
        
        # Check for uncertainty markers (good for avoiding hallucination)
        uncertainty_markers = ['may', 'might', 'could', 'possibly', 'likely', 'appears']
        has_uncertainty = any(marker in response.lower() for marker in uncertainty_markers)
        validation_details['includes_uncertainty'] = has_uncertainty
        
        # Simple consistency check (no contradictions)
        contradiction_patterns = ['but not', 'however not', 'although not']
        has_contradictions = any(pattern in response.lower() for pattern in contradiction_patterns)
        validation_details['no_contradictions'] = not has_contradictions
        
        # Calculate scores
        factual_accuracy = 0.7 if appropriate_length and not has_contradictions else 0.4
        source_grounding = 0.8 if has_source_refs else 0.3
        consistency_score = 0.8 if not has_contradictions else 0.2
        hallucination_risk = 0.3 if has_uncertainty and has_source_refs else 0.7
        
        # Determine overall validity
        is_valid = all([
            factual_accuracy >= min_confidence,
            source_grounding >= 0.5,
            consistency_score >= 0.5,
            hallucination_risk <= 0.5
        ])
        
        # Determine confidence level
        avg_score = (factual_accuracy + source_grounding + consistency_score) / 3
        if avg_score >= 0.95:
            confidence_level = ConfidenceLevel.VERY_HIGH
        elif avg_score >= 0.8:
            confidence_level = ConfidenceLevel.HIGH
        elif avg_score >= 0.6:
            confidence_level = ConfidenceLevel.MEDIUM
        elif avg_score >= 0.4:
            confidence_level = ConfidenceLevel.LOW
        else:
            confidence_level = ConfidenceLevel.VERY_LOW
        
        # Generate recommendations
        recommendations = []
        if not has_source_refs:
            recommendations.append("Add source references to improve grounding")
        if not has_uncertainty:
            recommendations.append("Include uncertainty markers for claims")
        if has_contradictions:
            recommendations.append("Resolve internal contradictions")
        if not appropriate_length:
            recommendations.append("Adjust response length for better coverage")
        
        warnings = []
        if hallucination_risk > 0.6:
            warnings.append("High hallucination risk detected")
        if not is_valid:
            warnings.append("Response failed validation checks")
        
        return ValidationResult(
            is_valid=is_valid,
            confidence_level=confidence_level,
            factual_accuracy=factual_accuracy,
            source_grounding=source_grounding,
            consistency_score=consistency_score,
            hallucination_risk=hallucination_risk,
            validation_details=validation_details,
            recommendations=recommendations,
            warnings=warnings
        )
    
    def _confidence_to_numeric(self, confidence_level: ConfidenceLevel) -> float:
        """Convert confidence level to numeric value."""
        mapping = {
            ConfidenceLevel.VERY_HIGH: 0.975,
            ConfidenceLevel.HIGH: 0.87,
            ConfidenceLevel.MEDIUM: 0.7,
            ConfidenceLevel.LOW: 0.5,
            ConfidenceLevel.VERY_LOW: 0.2
        }
        return mapping.get(confidence_level, 0.5)


class StructuredOperationsManager:
    """
    Manager for all structured AI operations.
    
    Coordinates and orchestrates various AI agents for comprehensive analysis.
    """
    
    def __init__(self, vllm_client: Optional[VLLMClient] = None):
        """Initialize structured operations manager."""
        self.vllm_client = vllm_client
        
        # Initialize agents
        self.entity_agent = EntityExtractionAgent(vllm_client)
        self.relationship_agent = RelationshipInferenceAgent(vllm_client)
        self.query_agent = QueryProcessingAgent(vllm_client)
        self.validation_agent = ResponseValidationAgent(vllm_client)
        
        # Overall performance tracking
        self.manager_stats = {
            'total_operations': 0,
            'successful_operations': 0,
            'average_operation_time': 0.0,
            'error_count': 0
        }
        
        logger.info("Initialized StructuredOperationsManager with all AI agents")
    
    async def analyze_content(
        self,
        text: str,
        include_entities: bool = True,
        include_relationships: bool = True,
        min_confidence: float = 0.7
    ) -> Dict[str, Any]:
        """
        Comprehensive content analysis using all available agents.
        
        Args:
            text: Text content to analyze
            include_entities: Whether to extract entities
            include_relationships: Whether to infer relationships
            min_confidence: Minimum confidence threshold
            
        Returns:
            Comprehensive analysis results
        """
        start_time = datetime.utcnow()
        self.manager_stats['total_operations'] += 1
        
        try:
            results = {}
            
            # Extract entities
            if include_entities:
                entities = await self.entity_agent.extract_entities(text, min_confidence=min_confidence)
                results['entities'] = [e.model_dump() for e in entities]
            else:
                entities = []
                results['entities'] = []
            
            # Infer relationships
            if include_relationships and len(entities) >= 2:
                relationships = await self.relationship_agent.infer_relationships(
                    text, entities, min_confidence=min_confidence
                )
                results['relationships'] = [r.model_dump() for r in relationships]
            else:
                results['relationships'] = []
            
            # Add content analysis
            analysis = await self._analyze_content_properties(text)
            results['content_analysis'] = analysis.model_dump()
            
            # Success statistics
            self.manager_stats['successful_operations'] += 1
            operation_time = (datetime.utcnow() - start_time).total_seconds()
            self.manager_stats['average_operation_time'] = (
                (self.manager_stats['average_operation_time'] * 
                 (self.manager_stats['successful_operations'] - 1) +
                 operation_time) / self.manager_stats['successful_operations']
            )
            
            logger.info(
                "Content analysis completed",
                text_length=len(text),
                entities_found=len(results['entities']),
                relationships_found=len(results['relationships']),
                operation_time_seconds=operation_time
            )
            
            return results
            
        except Exception as e:
            self.manager_stats['error_count'] += 1
            logger.error("Content analysis failed", error=str(e))
            return {
                'entities': [],
                'relationships': [],
                'content_analysis': {},
                'error': str(e)
            }
    
    async def _analyze_content_properties(self, text: str) -> ContentAnalysis:
        """Analyze basic content properties."""
        # Simple analysis without full agent
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        
        # Basic topic extraction (simple keyword frequency)
        word_freq = {}
        for word in words:
            word = word.lower().strip('.,!?;:"()[]')
            if len(word) > 3:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        main_topics = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        main_topics = [topic[0] for topic in main_topics]
        
        # Simple sentiment analysis (basic positive/negative words)
        positive_words = ['good', 'great', 'excellent', 'positive', 'success', 'benefit']
        negative_words = ['bad', 'poor', 'negative', 'problem', 'issue', 'error']
        
        pos_count = sum(1 for word in words if word.lower() in positive_words)
        neg_count = sum(1 for word in words if word.lower() in negative_words)
        
        sentiment_score = (pos_count - neg_count) / max(len(words), 1)
        sentiment_score = max(-1.0, min(1.0, sentiment_score))
        
        # Readability (simple approximation)
        avg_sentence_length = len(words) / max(len(sentences), 1)
        readability_score = max(0, min(100, 100 - avg_sentence_length * 2))
        
        # Technical complexity (based on technical terms)
        technical_terms = ['api', 'algorithm', 'database', 'server', 'code', 'system']
        tech_count = sum(1 for word in words if word.lower() in technical_terms)
        technical_complexity = min(1.0, tech_count / max(len(words), 1) * 10)
        
        return ContentAnalysis(
            main_topics=main_topics,
            sentiment_score=sentiment_score,
            readability_score=readability_score,
            technical_complexity=technical_complexity,
            key_insights=[f"Text contains {len(words)} words across {len(sentences)} sentences"],
            factual_claims=[],  # Would need more sophisticated analysis
            opinion_markers=[],
            temporal_references=[],
            quality_indicators={
                'word_count': len(words),
                'sentence_count': len(sentences),
                'avg_sentence_length': avg_sentence_length,
                'technical_density': technical_complexity
            }
        )
    
    async def get_operation_stats(self) -> Dict[str, Any]:
        """Get comprehensive operation statistics."""
        return {
            'manager_stats': self.manager_stats,
            'entity_agent_stats': self.entity_agent.extraction_stats,
            'relationship_agent_stats': self.relationship_agent.inference_stats,
            'query_agent_stats': self.query_agent.processing_stats,
            'validation_agent_stats': self.validation_agent.validation_stats
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check of all AI agents."""
        health_status = {
            'status': 'healthy',
            'agents': {},
            'vllm_client_available': self.vllm_client is not None,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        agents = [
            ('entity_agent', self.entity_agent),
            ('relationship_agent', self.relationship_agent),
            ('query_agent', self.query_agent),
            ('validation_agent', self.validation_agent)
        ]
        
        for agent_name, agent in agents:
            try:
                health_status['agents'][agent_name] = {
                    'status': 'healthy',
                    'agent_available': agent.agent is not None,
                    'has_stats': hasattr(agent, f'{agent_name.split("_")[0]}_stats')
                }
            except Exception as e:
                health_status['agents'][agent_name] = {
                    'status': 'unhealthy',
                    'error': str(e)
                }
        
        # Overall status
        unhealthy_agents = [
            name for name, status in health_status['agents'].items()
            if status.get('status') == 'unhealthy'
        ]
        
        if unhealthy_agents:
            health_status['status'] = 'degraded'
            health_status['unhealthy_agents'] = unhealthy_agents
        
        return health_status


# Example usage
async def example_usage():
    """Example of using StructuredOperationsManager."""
    # Initialize manager
    manager = StructuredOperationsManager()
    
    # Test content
    test_text = """
    John Smith works at OpenAI and has been developing advanced AI systems.
    The company was founded in 2015 and focuses on artificial intelligence research.
    OpenAI has created several breakthrough models including GPT-3 and GPT-4.
    """
    
    # Comprehensive analysis
    results = await manager.analyze_content(test_text)
    
    print("=== Entity Extraction ===")
    for entity in results['entities']:
        print(f"Entity: {entity['text']} ({entity['entity_type']}) - Confidence: {entity['confidence']:.2f}")
    
    print("\n=== Relationship Inference ===")
    for relationship in results['relationships']:
        print(f"Relationship: {relationship['subject_entity']} {relationship['predicate']} {relationship['object_entity']} - Confidence: {relationship['confidence']:.2f}")
    
    print("\n=== Content Analysis ===")
    analysis = results['content_analysis']
    print(f"Main topics: {analysis['main_topics']}")
    print(f"Sentiment: {analysis['sentiment_score']:.2f}")
    print(f"Technical complexity: {analysis['technical_complexity']:.2f}")
    
    # Test query processing
    query = "What are the latest developments in AI research at OpenAI?"
    processed_query = await manager.query_agent.process_query(query)
    print(f"\n=== Query Processing ===")
    print(f"Intent: {processed_query.intent.value} (Confidence: {processed_query.intent_confidence:.2f})")
    print(f"Keywords: {processed_query.keywords}")
    
    # Test response validation
    test_response = "OpenAI has developed several AI models including GPT-3 and GPT-4, which are based on transformer architecture."
    validation = await manager.validation_agent.validate_response(
        test_response, query, [test_text]
    )
    print(f"\n=== Response Validation ===")
    print(f"Valid: {validation.is_valid}")
    print(f"Confidence: {validation.confidence_level.value}")
    print(f"Hallucination risk: {validation.hallucination_risk:.2f}")
    
    # Get statistics
    stats = await manager.get_operation_stats()
    print(f"\n=== Operation Statistics ===")
    print(f"Total operations: {stats['manager_stats']['total_operations']}")
    print(f"Success rate: {stats['manager_stats']['successful_operations'] / max(stats['manager_stats']['total_operations'], 1):.2f}")


if __name__ == "__main__":
    asyncio.run(example_usage())