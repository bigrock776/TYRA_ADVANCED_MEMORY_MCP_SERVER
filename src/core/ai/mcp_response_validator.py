"""
MCP Tool Response Validation with Pydantic AI.

This module provides comprehensive validation for MCP tool responses,
structured error handling, performance metrics validation, and tool input
validation using Pydantic schemas for maximum reliability.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple, Type
from dataclasses import dataclass
from enum import Enum
import hashlib

import structlog
from pydantic import BaseModel, Field, ConfigDict, ValidationError, field_validator
from pydantic_ai import Agent, RunContext

from mcp.types import CallToolResult, TextContent, ImageContent, EmbeddedResource
from ..clients.vllm_client import VLLMClient
from ..utils.config import settings

logger = structlog.get_logger(__name__)


class ValidationSeverity(str, Enum):
    """Severity levels for validation issues."""
    CRITICAL = "critical"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class ResponseType(str, Enum):
    """Types of MCP tool responses."""
    MEMORY_OPERATION = "memory_operation"
    SEARCH_RESULT = "search_result"
    ANALYTICS_DATA = "analytics_data"
    SYNTHESIS_RESULT = "synthesis_result"
    ERROR_RESPONSE = "error_response"
    SYSTEM_STATUS = "system_status"


class ValidationResult(BaseModel):
    """Structured validation result for MCP responses."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    is_valid: bool = Field(..., description="Overall validation result")
    validation_score: float = Field(..., ge=0.0, le=1.0, description="Validation score")
    response_type: ResponseType = Field(..., description="Type of response validated")
    issues: List[Dict[str, Any]] = Field(default_factory=list, description="Validation issues found")
    recommendations: List[str] = Field(default_factory=list, description="Improvement recommendations")
    performance_metrics: Dict[str, float] = Field(default_factory=dict, description="Performance metrics")
    validation_time_ms: float = Field(..., ge=0.0, description="Validation time in milliseconds")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Validation timestamp")


class MemoryOperationResponse(BaseModel):
    """Validated memory operation response schema."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    success: bool = Field(..., description="Operation success status")
    operation_type: str = Field(..., min_length=1, description="Type of memory operation")
    memory_id: Optional[str] = Field(None, description="Memory identifier if applicable")
    affected_count: int = Field(..., ge=0, description="Number of memories affected")
    execution_time_ms: float = Field(..., ge=0.0, description="Execution time in milliseconds")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Operation metadata")
    
    @field_validator('operation_type')
    @classmethod
    def validate_operation_type(cls, v):
        """Validate operation type is recognized."""
        valid_types = {'store', 'retrieve', 'update', 'delete', 'search', 'analyze'}
        if v.lower() not in valid_types:
            raise ValueError(f"Unknown operation type: {v}")
        return v.lower()


class SearchResultResponse(BaseModel):
    """Validated search result response schema."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    results: List[Dict[str, Any]] = Field(..., description="Search results")
    total_found: int = Field(..., ge=0, description="Total results found")
    query_time_ms: float = Field(..., ge=0.0, description="Query execution time")
    confidence_scores: List[float] = Field(..., description="Confidence scores for results")
    search_strategy: str = Field(..., description="Search strategy used")
    hallucination_risk: float = Field(..., ge=0.0, le=1.0, description="Hallucination risk assessment")
    
    @field_validator('confidence_scores')
    @classmethod
    def validate_confidence_scores(cls, v, info):
        """Ensure confidence scores match results count."""
        if 'results' in info.data and len(v) != len(info.data['results']):
            raise ValueError("Confidence scores count must match results count")
        if any(score < 0.0 or score > 1.0 for score in v):
            raise ValueError("All confidence scores must be between 0.0 and 1.0")
        return v


class AnalyticsDataResponse(BaseModel):
    """Validated analytics data response schema."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    metrics: Dict[str, Union[float, int]] = Field(..., description="Analytics metrics")
    time_period: str = Field(..., description="Time period for metrics")
    data_points: int = Field(..., ge=0, description="Number of data points analyzed")
    accuracy_estimate: float = Field(..., ge=0.0, le=1.0, description="Data accuracy estimate")
    insights: List[str] = Field(default_factory=list, description="Generated insights")
    
    @field_validator('metrics')
    @classmethod
    def validate_metrics(cls, v):
        """Validate metrics contain expected keys."""
        required_keys = {'total_operations', 'success_rate', 'average_response_time'}
        if not all(key in v for key in required_keys):
            raise ValueError(f"Metrics must contain: {required_keys}")
        return v


class ErrorResponse(BaseModel):
    """Validated error response schema."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    error_type: str = Field(..., min_length=1, description="Type of error")
    error_message: str = Field(..., min_length=1, description="Human-readable error message")
    error_code: Optional[str] = Field(None, description="Error code if applicable")
    details: Dict[str, Any] = Field(default_factory=dict, description="Additional error details")
    recovery_suggestions: List[str] = Field(default_factory=list, description="Recovery suggestions")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")


class MCPResponseValidator:
    """
    Comprehensive MCP tool response validator.
    
    Validates MCP tool responses using Pydantic schemas with AI-enhanced validation
    for content quality, consistency, and reliability.
    """
    
    def __init__(self, vllm_client: Optional[VLLMClient] = None):
        """Initialize MCP response validator."""
        self.vllm_client = vllm_client
        self.validation_agent = self._create_validation_agent()
        
        # Response schemas by type
        self.response_schemas = {
            ResponseType.MEMORY_OPERATION: MemoryOperationResponse,
            ResponseType.SEARCH_RESULT: SearchResultResponse,
            ResponseType.ANALYTICS_DATA: AnalyticsDataResponse,
            ResponseType.ERROR_RESPONSE: ErrorResponse
        }
        
        # Validation statistics
        self.validation_stats = {
            'total_validations': 0,
            'successful_validations': 0,
            'failed_validations': 0,
            'average_validation_time': 0.0,
            'validation_by_type': {rt.value: 0 for rt in ResponseType},
            'common_issues': {}
        }
        
        logger.info("Initialized MCPResponseValidator")
    
    def _create_validation_agent(self) -> Optional[Agent]:
        """Create Pydantic AI agent for advanced response validation."""
        try:
            if not self.vllm_client:
                logger.warning("No VLLMClient provided, using basic validation only")
                return None
            
            from pydantic_ai.models.openai import OpenAIModel
            
            model = OpenAIModel(
                'llama2',
                base_url='http://localhost:8000/v1',
                api_key='not-needed'
            )
            
            return Agent(
                model,
                result_type=ValidationResult,
                system_prompt="""
                You are an expert MCP tool response validator. Your task is to validate
                tool responses for correctness, completeness, and reliability.
                
                Instructions:
                1. Check response structure and required fields
                2. Validate data types and value ranges
                3. Assess content quality and consistency
                4. Identify potential issues or anomalies
                5. Provide specific recommendations for improvement
                6. Calculate accurate validation scores
                
                Focus on ensuring responses are reliable, complete, and useful.
                Be thorough in identifying issues and providing actionable feedback.
                """,
                retries=2
            )
            
        except Exception as e:
            logger.error("Failed to create validation agent", error=str(e))
            return None
    
    async def validate_response(
        self,
        response: CallToolResult,
        expected_type: ResponseType,
        tool_name: str,
        execution_context: Optional[Dict[str, Any]] = None
    ) -> ValidationResult:
        """
        Comprehensive validation of MCP tool response.
        
        Args:
            response: MCP tool response to validate
            expected_type: Expected response type
            tool_name: Name of the tool that generated response
            execution_context: Optional execution context for validation
            
        Returns:
            Comprehensive validation result
        """
        start_time = datetime.utcnow()
        self.validation_stats['total_validations'] += 1
        self.validation_stats['validation_by_type'][expected_type.value] += 1
        
        try:
            # Extract response content
            response_content = self._extract_response_content(response)
            
            # Basic structure validation
            structure_validation = await self._validate_response_structure(
                response_content, expected_type
            )
            
            # Schema validation
            schema_validation = await self._validate_response_schema(
                response_content, expected_type
            )
            
            # Content quality validation
            quality_validation = await self._validate_content_quality(
                response_content, tool_name, execution_context
            )
            
            # Performance metrics validation
            performance_validation = await self._validate_performance_metrics(
                response_content, execution_context
            )
            
            # AI-enhanced validation if available
            ai_validation = None
            if self.validation_agent:
                ai_validation = await self._ai_enhanced_validation(
                    response_content, expected_type, tool_name
                )
            
            # Combine validation results
            validation_result = await self._combine_validation_results(
                structure_validation,
                schema_validation,
                quality_validation,
                performance_validation,
                ai_validation,
                expected_type
            )
            
            # Update statistics
            validation_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            validation_result.validation_time_ms = validation_time
            
            if validation_result.is_valid:
                self.validation_stats['successful_validations'] += 1
            else:
                self.validation_stats['failed_validations'] += 1
                
                # Track common issues
                for issue in validation_result.issues:
                    issue_type = issue.get('type', 'unknown')
                    self.validation_stats['common_issues'][issue_type] = (
                        self.validation_stats['common_issues'].get(issue_type, 0) + 1
                    )
            
            self.validation_stats['average_validation_time'] = (
                (self.validation_stats['average_validation_time'] * 
                 (self.validation_stats['total_validations'] - 1) + validation_time) /
                self.validation_stats['total_validations']
            )
            
            logger.info(
                "Response validation completed",
                tool_name=tool_name,
                response_type=expected_type.value,
                is_valid=validation_result.is_valid,
                validation_score=validation_result.validation_score,
                validation_time_ms=validation_time
            )
            
            return validation_result
            
        except Exception as e:
            logger.error("Response validation failed", error=str(e))
            return ValidationResult(
                is_valid=False,
                validation_score=0.0,
                response_type=expected_type,
                issues=[{
                    'type': 'validation_error',
                    'severity': ValidationSeverity.CRITICAL.value,
                    'message': f"Validation failed: {str(e)}"
                }],
                recommendations=["Manual review required due to validation error"],
                validation_time_ms=(datetime.utcnow() - start_time).total_seconds() * 1000
            )
    
    def _extract_response_content(self, response: CallToolResult) -> Dict[str, Any]:
        """Extract content from MCP response."""
        try:
            if response.content and len(response.content) > 0:
                content_item = response.content[0]
                
                if hasattr(content_item, 'text'):
                    # Try to parse as JSON
                    try:
                        return json.loads(content_item.text)
                    except json.JSONDecodeError:
                        return {'text': content_item.text}
                else:
                    return {'content': str(content_item)}
            
            return {}
            
        except Exception as e:
            logger.warning("Failed to extract response content", error=str(e))
            return {}
    
    async def _validate_response_structure(
        self,
        content: Dict[str, Any],
        expected_type: ResponseType
    ) -> Dict[str, Any]:
        """Validate basic response structure."""
        issues = []
        score = 1.0
        
        # Check if content is empty
        if not content:
            issues.append({
                'type': 'empty_response',
                'severity': ValidationSeverity.CRITICAL.value,
                'message': 'Response content is empty'
            })
            score = 0.0
        
        # Check for required fields based on response type
        required_fields = self._get_required_fields(expected_type)
        for field in required_fields:
            if field not in content:
                issues.append({
                    'type': 'missing_field',
                    'severity': ValidationSeverity.ERROR.value,
                    'message': f'Required field missing: {field}',
                    'field': field
                })
                score -= 0.2
        
        return {
            'structure_score': max(0.0, score),
            'structure_issues': issues
        }
    
    async def _validate_response_schema(
        self,
        content: Dict[str, Any],
        expected_type: ResponseType
    ) -> Dict[str, Any]:
        """Validate response against Pydantic schema."""
        issues = []
        score = 1.0
        
        try:
            if expected_type in self.response_schemas:
                schema_class = self.response_schemas[expected_type]
                
                # Attempt to validate with schema
                try:
                    validated_response = schema_class(**content)
                    score = 1.0
                except ValidationError as e:
                    for error in e.errors():
                        issues.append({
                            'type': 'schema_validation',
                            'severity': ValidationSeverity.ERROR.value,
                            'message': f"Schema validation error: {error['msg']}",
                            'field': '.'.join(str(loc) for loc in error['loc']),
                            'error_type': error['type']
                        })
                        score -= 0.3
            else:
                issues.append({
                    'type': 'unknown_response_type',
                    'severity': ValidationSeverity.WARNING.value,
                    'message': f'No schema available for response type: {expected_type.value}'
                })
                score = 0.7
        
        except Exception as e:
            issues.append({
                'type': 'schema_validation_error',
                'severity': ValidationSeverity.ERROR.value,
                'message': f'Schema validation failed: {str(e)}'
            })
            score = 0.3
        
        return {
            'schema_score': max(0.0, score),
            'schema_issues': issues
        }
    
    async def _validate_content_quality(
        self,
        content: Dict[str, Any],
        tool_name: str,
        execution_context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Validate content quality and completeness."""
        issues = []
        score = 1.0
        
        # Check for completeness
        if isinstance(content, dict):
            # Check for empty values
            empty_fields = [k for k, v in content.items() if v == "" or v is None]
            if empty_fields:
                issues.append({
                    'type': 'empty_fields',
                    'severity': ValidationSeverity.WARNING.value,
                    'message': f'Empty fields detected: {empty_fields}',
                    'fields': empty_fields
                })
                score -= 0.1 * len(empty_fields)
            
            # Check for reasonable data sizes
            for key, value in content.items():
                if isinstance(value, str) and len(value) > 10000:
                    issues.append({
                        'type': 'oversized_field',
                        'severity': ValidationSeverity.WARNING.value,
                        'message': f'Field {key} is unusually large ({len(value)} characters)',
                        'field': key,
                        'size': len(value)
                    })
                    score -= 0.05
        
        # Tool-specific validation
        tool_issues, tool_score = await self._validate_tool_specific_content(
            content, tool_name, execution_context
        )
        issues.extend(tool_issues)
        score = min(score, tool_score)
        
        return {
            'quality_score': max(0.0, score),
            'quality_issues': issues
        }
    
    async def _validate_performance_metrics(
        self,
        content: Dict[str, Any],
        execution_context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Validate performance metrics in response."""
        issues = []
        score = 1.0
        metrics = {}
        
        # Check execution time if present
        if 'execution_time_ms' in content:
            exec_time = content['execution_time_ms']
            metrics['execution_time_ms'] = exec_time
            
            if exec_time > 5000:  # 5 seconds
                issues.append({
                    'type': 'slow_execution',
                    'severity': ValidationSeverity.WARNING.value,
                    'message': f'Slow execution time: {exec_time}ms',
                    'execution_time': exec_time
                })
                score -= 0.1
            elif exec_time > 10000:  # 10 seconds
                issues.append({
                    'type': 'very_slow_execution',
                    'severity': ValidationSeverity.ERROR.value,
                    'message': f'Very slow execution time: {exec_time}ms',
                    'execution_time': exec_time
                })
                score -= 0.3
        
        # Check result counts vs performance
        if 'total_found' in content and 'query_time_ms' in content:
            total_found = content['total_found']
            query_time = content['query_time_ms']
            
            if total_found > 0:
                time_per_result = query_time / total_found
                metrics['time_per_result_ms'] = time_per_result
                
                if time_per_result > 100:  # 100ms per result
                    issues.append({
                        'type': 'inefficient_retrieval',
                        'severity': ValidationSeverity.WARNING.value,
                        'message': f'Inefficient retrieval: {time_per_result:.2f}ms per result',
                        'time_per_result': time_per_result
                    })
                    score -= 0.1
        
        return {
            'performance_score': max(0.0, score),
            'performance_issues': issues,
            'performance_metrics': metrics
        }
    
    async def _ai_enhanced_validation(
        self,
        content: Dict[str, Any],
        expected_type: ResponseType,
        tool_name: str
    ) -> Optional[Dict[str, Any]]:
        """AI-enhanced validation using Pydantic AI agent."""
        if not self.validation_agent:
            return None
        
        try:
            # Prepare validation context
            validation_context = {
                'content': content,
                'expected_type': expected_type.value,
                'tool_name': tool_name,
                'content_size': len(str(content))
            }
            
            result = await self.validation_agent.run(
                f"Validate this MCP tool response: {json.dumps(content, default=str)[:1000]}",
                deps=validation_context
            )
            
            ai_validation = result.data
            return {
                'ai_score': ai_validation.validation_score,
                'ai_issues': [issue for issue in ai_validation.issues],
                'ai_recommendations': ai_validation.recommendations
            }
            
        except Exception as e:
            logger.warning("AI-enhanced validation failed", error=str(e))
            return None
    
    async def _validate_tool_specific_content(
        self,
        content: Dict[str, Any],
        tool_name: str,
        execution_context: Optional[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], float]:
        """Validate content specific to the tool type."""
        issues = []
        score = 1.0
        
        # Memory operation tools
        if tool_name in ['store_memory', 'retrieve_memory', 'search_memories']:
            if tool_name == 'search_memories':
                # Validate search results
                if 'results' in content:
                    results = content['results']
                    if not isinstance(results, list):
                        issues.append({
                            'type': 'invalid_results_format',
                            'severity': ValidationSeverity.ERROR.value,
                            'message': 'Search results must be a list'
                        })
                        score -= 0.5
                    elif len(results) == 0 and content.get('total_found', 0) > 0:
                        issues.append({
                            'type': 'inconsistent_result_count',
                            'severity': ValidationSeverity.WARNING.value,
                            'message': 'total_found > 0 but results list is empty'
                        })
                        score -= 0.2
            
            elif tool_name == 'store_memory':
                # Validate memory storage
                if 'memory_id' not in content:
                    issues.append({
                        'type': 'missing_memory_id',
                        'severity': ValidationSeverity.ERROR.value,
                        'message': 'Memory storage should return memory_id'
                    })
                    score -= 0.4
        
        # Analytics tools
        elif tool_name in ['get_analytics', 'get_performance_metrics']:
            if 'metrics' in content:
                metrics = content['metrics']
                if not isinstance(metrics, dict):
                    issues.append({
                        'type': 'invalid_metrics_format',
                        'severity': ValidationSeverity.ERROR.value,
                        'message': 'Metrics must be a dictionary'
                    })
                    score -= 0.5
                elif len(metrics) == 0:
                    issues.append({
                        'type': 'empty_metrics',
                        'severity': ValidationSeverity.WARNING.value,
                        'message': 'No metrics data returned'
                    })
                    score -= 0.3
        
        return issues, max(0.0, score)
    
    async def _combine_validation_results(
        self,
        structure_validation: Dict[str, Any],
        schema_validation: Dict[str, Any],
        quality_validation: Dict[str, Any],
        performance_validation: Dict[str, Any],
        ai_validation: Optional[Dict[str, Any]],
        response_type: ResponseType
    ) -> ValidationResult:
        """Combine all validation results into final assessment."""
        
        # Collect all issues
        all_issues = []
        all_issues.extend(structure_validation.get('structure_issues', []))
        all_issues.extend(schema_validation.get('schema_issues', []))
        all_issues.extend(quality_validation.get('quality_issues', []))
        all_issues.extend(performance_validation.get('performance_issues', []))
        
        if ai_validation:
            all_issues.extend(ai_validation.get('ai_issues', []))
        
        # Calculate overall score (weighted average)
        weights = {
            'structure': 0.3,
            'schema': 0.3,
            'quality': 0.2,
            'performance': 0.1,
            'ai': 0.1 if ai_validation else 0.0
        }
        
        if not ai_validation:
            # Redistribute AI weight
            weights['structure'] += 0.05
            weights['quality'] += 0.05
        
        overall_score = (
            structure_validation.get('structure_score', 0.0) * weights['structure'] +
            schema_validation.get('schema_score', 0.0) * weights['schema'] +
            quality_validation.get('quality_score', 0.0) * weights['quality'] +
            performance_validation.get('performance_score', 0.0) * weights['performance']
        )
        
        if ai_validation:
            overall_score += ai_validation.get('ai_score', 0.0) * weights['ai']
        
        # Determine if response is valid (no critical issues and score > 0.6)
        critical_issues = [issue for issue in all_issues 
                          if issue.get('severity') == ValidationSeverity.CRITICAL.value]
        is_valid = len(critical_issues) == 0 and overall_score >= 0.6
        
        # Generate recommendations
        recommendations = []
        if not is_valid:
            recommendations.append("Response requires attention before use")
        
        if overall_score < 0.8:
            recommendations.append("Consider optimizing response quality")
        
        if any(issue.get('type') == 'slow_execution' for issue in all_issues):
            recommendations.append("Optimize execution performance")
        
        if ai_validation:
            recommendations.extend(ai_validation.get('ai_recommendations', []))
        
        # Collect performance metrics
        performance_metrics = performance_validation.get('performance_metrics', {})
        performance_metrics['overall_score'] = overall_score
        
        return ValidationResult(
            is_valid=is_valid,
            validation_score=overall_score,
            response_type=response_type,
            issues=all_issues,
            recommendations=recommendations,
            performance_metrics=performance_metrics,
            validation_time_ms=0.0  # Will be set by caller
        )
    
    def _get_required_fields(self, response_type: ResponseType) -> List[str]:
        """Get required fields for response type."""
        field_map = {
            ResponseType.MEMORY_OPERATION: ['success', 'operation_type'],
            ResponseType.SEARCH_RESULT: ['results', 'total_found'],
            ResponseType.ANALYTICS_DATA: ['metrics'],
            ResponseType.ERROR_RESPONSE: ['error_type', 'error_message'],
            ResponseType.SYSTEM_STATUS: ['status']
        }
        return field_map.get(response_type, [])
    
    async def get_validation_stats(self) -> Dict[str, Any]:
        """Get comprehensive validation statistics."""
        return {
            **self.validation_stats,
            'success_rate': (
                self.validation_stats['successful_validations'] /
                max(self.validation_stats['total_validations'], 1)
            ),
            'failure_rate': (
                self.validation_stats['failed_validations'] /
                max(self.validation_stats['total_validations'], 1)
            )
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check of validator components."""
        health_status = {
            'status': 'healthy',
            'validation_agent_available': self.validation_agent is not None,
            'vllm_client_available': self.vllm_client is not None,
            'schemas_loaded': len(self.response_schemas),
            'total_validations': self.validation_stats['total_validations'],
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Test validation agent if available
        if self.validation_agent:
            try:
                # Simple test validation
                test_result = await self.validation_agent.run(
                    "Test validation check",
                    deps={'test': True}
                )
                health_status['validation_agent_status'] = 'healthy'
            except Exception as e:
                health_status['validation_agent_status'] = 'unhealthy'
                health_status['validation_agent_error'] = str(e)
        
        return health_status


# Example usage
async def example_usage():
    """Example of using MCPResponseValidator."""
    # Initialize validator
    validator = MCPResponseValidator()
    
    # Mock MCP response
    from mcp.types import CallToolResult, TextContent
    
    mock_response = CallToolResult(
        content=[TextContent(
            type="text",
            text=json.dumps({
                "success": True,
                "operation_type": "store",
                "memory_id": "mem_12345",
                "affected_count": 1,
                "execution_time_ms": 45.2,
                "metadata": {"source": "test"}
            })
        )]
    )
    
    # Validate response
    validation_result = await validator.validate_response(
        mock_response,
        ResponseType.MEMORY_OPERATION,
        "store_memory"
    )
    
    print(f"Validation Result:")
    print(f"  Valid: {validation_result.is_valid}")
    print(f"  Score: {validation_result.validation_score:.2f}")
    print(f"  Issues: {len(validation_result.issues)}")
    print(f"  Recommendations: {validation_result.recommendations}")
    
    # Get statistics
    stats = await validator.get_validation_stats()
    print(f"Validation stats: {stats}")


if __name__ == "__main__":
    asyncio.run(example_usage())