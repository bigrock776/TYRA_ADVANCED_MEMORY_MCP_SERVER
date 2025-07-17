"""
Event-Driven Trigger System for Automated Memory Management.

This module provides a comprehensive rule-based trigger system for automating
memory operations, notifications, and custom workflows. All processing is 
performed locally with sandboxed Python execution for security.
"""

import asyncio
import ast
import operator
import time
import json
import re
from typing import Dict, List, Optional, Set, Any, Callable, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import uuid
import math

# Safe execution imports
import builtins
from types import CodeType

import structlog
from pydantic import BaseModel, Field, ConfigDict, validator

from ...models.memory import Memory
from ...core.cache.redis_cache import RedisCache
from ...core.utils.config import settings

logger = structlog.get_logger(__name__)


class TriggerType(str, Enum):
    """Types of triggers supported."""
    MEMORY_CREATED = "memory_created"
    MEMORY_UPDATED = "memory_updated"
    MEMORY_DELETED = "memory_deleted"
    MEMORY_ACCESSED = "memory_accessed"
    SEARCH_PERFORMED = "search_performed"
    USER_LOGIN = "user_login"
    USER_LOGOUT = "user_logout"
    SYSTEM_EVENT = "system_event"
    SCHEDULED = "scheduled"
    CUSTOM = "custom"


class ConditionOperator(str, Enum):
    """Operators for condition evaluation."""
    EQUALS = "eq"
    NOT_EQUALS = "ne"
    GREATER_THAN = "gt"
    GREATER_EQUAL = "ge"
    LESS_THAN = "lt"
    LESS_EQUAL = "le"
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"
    REGEX_MATCH = "regex_match"
    IN_LIST = "in"
    NOT_IN_LIST = "not_in"
    IS_NULL = "is_null"
    IS_NOT_NULL = "is_not_null"


class ActionType(str, Enum):
    """Types of actions that can be executed."""
    SEND_NOTIFICATION = "send_notification"
    CREATE_MEMORY = "create_memory"
    UPDATE_MEMORY = "update_memory"
    DELETE_MEMORY = "delete_memory"
    ARCHIVE_MEMORY = "archive_memory"
    TAG_MEMORY = "tag_memory"
    EXECUTE_SEARCH = "execute_search"
    CALL_WEBHOOK = "call_webhook"
    LOG_EVENT = "log_event"
    CUSTOM_FUNCTION = "custom_function"


@dataclass
class TriggerCondition:
    """Represents a condition that must be met for trigger execution."""
    field: str
    operator: ConditionOperator
    value: Any
    case_sensitive: bool = True
    
    def evaluate(self, context: Dict[str, Any]) -> bool:
        """Evaluate the condition against the provided context."""
        try:
            # Get field value from context using dot notation
            field_value = self._get_field_value(context, self.field)
            
            # Handle null checks first
            if self.operator == ConditionOperator.IS_NULL:
                return field_value is None
            elif self.operator == ConditionOperator.IS_NOT_NULL:
                return field_value is not None
            
            # Skip other checks if field is None
            if field_value is None:
                return False
            
            # Apply case sensitivity for string operations
            if isinstance(field_value, str) and not self.case_sensitive:
                field_value = field_value.lower()
            if isinstance(self.value, str) and not self.case_sensitive:
                compare_value = self.value.lower()
            else:
                compare_value = self.value
            
            # Evaluate based on operator
            if self.operator == ConditionOperator.EQUALS:
                return field_value == compare_value
            elif self.operator == ConditionOperator.NOT_EQUALS:
                return field_value != compare_value
            elif self.operator == ConditionOperator.GREATER_THAN:
                return field_value > compare_value
            elif self.operator == ConditionOperator.GREATER_EQUAL:
                return field_value >= compare_value
            elif self.operator == ConditionOperator.LESS_THAN:
                return field_value < compare_value
            elif self.operator == ConditionOperator.LESS_EQUAL:
                return field_value <= compare_value
            elif self.operator == ConditionOperator.CONTAINS:
                return str(compare_value) in str(field_value)
            elif self.operator == ConditionOperator.NOT_CONTAINS:
                return str(compare_value) not in str(field_value)
            elif self.operator == ConditionOperator.STARTS_WITH:
                return str(field_value).startswith(str(compare_value))
            elif self.operator == ConditionOperator.ENDS_WITH:
                return str(field_value).endswith(str(compare_value))
            elif self.operator == ConditionOperator.REGEX_MATCH:
                return bool(re.search(str(compare_value), str(field_value)))
            elif self.operator == ConditionOperator.IN_LIST:
                return field_value in compare_value if isinstance(compare_value, (list, tuple, set)) else False
            elif self.operator == ConditionOperator.NOT_IN_LIST:
                return field_value not in compare_value if isinstance(compare_value, (list, tuple, set)) else True
            
            return False
            
        except Exception as e:
            logger.warning(f"Error evaluating condition: {e}")
            return False
    
    def _get_field_value(self, context: Dict[str, Any], field_path: str) -> Any:
        """Get nested field value using dot notation."""
        try:
            value = context
            for part in field_path.split('.'):
                if isinstance(value, dict):
                    value = value.get(part)
                elif hasattr(value, part):
                    value = getattr(value, part)
                else:
                    return None
            return value
        except:
            return None


@dataclass
class TriggerAction:
    """Represents an action to be executed when trigger conditions are met."""
    action_type: ActionType
    parameters: Dict[str, Any] = field(default_factory=dict)
    delay_seconds: float = 0.0
    retry_count: int = 0
    retry_delay: float = 1.0
    
    async def execute(self, context: Dict[str, Any], action_handlers: Dict[ActionType, Callable]) -> bool:
        """Execute the action with the provided context."""
        try:
            # Apply delay if specified
            if self.delay_seconds > 0:
                await asyncio.sleep(self.delay_seconds)
            
            # Get action handler
            handler = action_handlers.get(self.action_type)
            if not handler:
                logger.warning(f"No handler found for action type: {self.action_type}")
                return False
            
            # Execute with retries
            for attempt in range(self.retry_count + 1):
                try:
                    # Merge context and parameters
                    execution_params = {**context, **self.parameters}
                    
                    # Execute handler
                    if asyncio.iscoroutinefunction(handler):
                        result = await handler(execution_params)
                    else:
                        result = handler(execution_params)
                    
                    # Consider None or True as success
                    return result is not False
                    
                except Exception as e:
                    if attempt < self.retry_count:
                        logger.warning(f"Action attempt {attempt + 1} failed, retrying: {e}")
                        await asyncio.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
                    else:
                        logger.error(f"Action failed after {attempt + 1} attempts: {e}")
                        return False
            
            return False
            
        except Exception as e:
            logger.error(f"Error executing action: {e}")
            return False


@dataclass
class TriggerRule:
    """Represents a complete trigger rule with conditions and actions."""
    id: str
    name: str
    trigger_type: TriggerType
    conditions: List[TriggerCondition]
    actions: List[TriggerAction]
    enabled: bool = True
    condition_logic: str = "AND"  # "AND" or "OR"
    max_executions: Optional[int] = None
    execution_count: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_executed: Optional[datetime] = None
    
    def evaluate_conditions(self, context: Dict[str, Any]) -> bool:
        """Evaluate all conditions according to the logic operator."""
        if not self.conditions:
            return True
        
        results = [condition.evaluate(context) for condition in self.conditions]
        
        if self.condition_logic.upper() == "OR":
            return any(results)
        else:  # Default to AND
            return all(results)
    
    def can_execute(self) -> bool:
        """Check if the rule can be executed."""
        if not self.enabled:
            return False
        
        if self.max_executions and self.execution_count >= self.max_executions:
            return False
        
        return True
    
    def record_execution(self):
        """Record that the rule was executed."""
        self.execution_count += 1
        self.last_executed = datetime.utcnow()


class Trigger(BaseModel):
    """Pydantic model for trigger configuration."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    id: str = Field(description="Unique trigger ID")
    name: str = Field(description="Human-readable trigger name")
    trigger_type: TriggerType = Field(description="Type of trigger")
    conditions: List[Dict[str, Any]] = Field(default_factory=list, description="Trigger conditions")
    actions: List[Dict[str, Any]] = Field(default_factory=list, description="Actions to execute")
    enabled: bool = Field(default=True, description="Whether trigger is enabled")
    condition_logic: str = Field(default="AND", description="Logic for combining conditions")
    max_executions: Optional[int] = Field(default=None, description="Maximum executions")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    @validator('condition_logic')
    def validate_condition_logic(cls, v):
        if v.upper() not in ['AND', 'OR']:
            raise ValueError('condition_logic must be "AND" or "OR"')
        return v.upper()


@dataclass
class ExecutionContext:
    """Context for trigger execution."""
    trigger_type: TriggerType
    event_data: Dict[str, Any]
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TriggerResult:
    """Result of trigger execution."""
    trigger_id: str
    success: bool
    execution_time: float
    actions_executed: int
    actions_successful: int
    error_message: Optional[str] = None
    context: Optional[Dict[str, Any]] = None


class SafeEvaluator:
    """Safe evaluator for custom Python expressions."""
    
    # Allowed built-in functions
    SAFE_BUILTINS = {
        'abs', 'all', 'any', 'bool', 'dict', 'enumerate', 'filter', 'float',
        'int', 'len', 'list', 'map', 'max', 'min', 'range', 'round', 'set',
        'sorted', 'str', 'sum', 'tuple', 'zip'
    }
    
    # Allowed operators
    SAFE_OPERATORS = {
        ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow, ast.FloorDiv,
        ast.LShift, ast.RShift, ast.BitOr, ast.BitXor, ast.BitAnd,
        ast.And, ast.Or, ast.Not, ast.Invert, ast.UAdd, ast.USub,
        ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
        ast.Is, ast.IsNot, ast.In, ast.NotIn
    }
    
    # Allowed node types
    SAFE_NODES = {
        ast.Expression, ast.BinOp, ast.UnaryOp, ast.Compare, ast.BoolOp,
        ast.Constant, ast.Num, ast.Str, ast.List, ast.Tuple, ast.Dict,
        ast.Name, ast.Load, ast.Store, ast.Subscript, ast.Index,
        ast.Attribute, ast.Call, ast.keyword
    }
    
    def __init__(self):
        self.safe_names = {
            '__builtins__': {name: getattr(builtins, name) for name in self.SAFE_BUILTINS}
        }
    
    def is_safe(self, node: ast.AST) -> bool:
        """Check if AST node is safe for execution."""
        if type(node) not in self.SAFE_NODES:
            return False
        
        if isinstance(node, ast.Call):
            # Only allow calls to safe built-in functions
            if isinstance(node.func, ast.Name):
                return node.func.id in self.SAFE_BUILTINS
            return False
        
        if isinstance(node, ast.Attribute):
            # Allow some safe attribute access
            safe_attrs = {'upper', 'lower', 'strip', 'split', 'join', 'startswith', 'endswith'}
            return node.attr in safe_attrs
        
        if isinstance(node, (ast.BinOp, ast.UnaryOp, ast.Compare, ast.BoolOp)):
            op_type = type(node.op) if hasattr(node, 'op') else None
            if hasattr(node, 'ops'):  # Compare nodes
                return all(type(op) in self.SAFE_OPERATORS for op in node.ops)
            return op_type in self.SAFE_OPERATORS
        
        # Recursively check child nodes
        for child in ast.iter_child_nodes(node):
            if not self.is_safe(child):
                return False
        
        return True
    
    def evaluate(self, expression: str, context: Dict[str, Any]) -> Any:
        """Safely evaluate a Python expression."""
        try:
            # Parse the expression
            tree = ast.parse(expression, mode='eval')
            
            # Check if it's safe
            if not self.is_safe(tree):
                raise ValueError("Unsafe expression")
            
            # Compile and execute
            code = compile(tree, '<string>', 'eval')
            safe_context = {**self.safe_names, **context}
            
            return eval(code, safe_context)
            
        except Exception as e:
            logger.warning(f"Error evaluating expression '{expression}': {e}")
            raise


class TriggerEngine:
    """
    Main trigger engine for event-driven automation.
    
    Features:
    - Rule-based triggers with flexible conditions
    - Custom Python expressions with sandboxed execution
    - Asynchronous action execution with retries
    - Performance monitoring and statistics
    - Memory-efficient event processing
    - Comprehensive logging and debugging
    """
    
    def __init__(
        self,
        redis_cache: Optional[RedisCache] = None,
        max_concurrent_executions: int = 50,
        execution_timeout: float = 30.0
    ):
        """
        Initialize the trigger engine.
        
        Args:
            redis_cache: Optional Redis cache for trigger persistence
            max_concurrent_executions: Maximum concurrent trigger executions
            execution_timeout: Timeout for trigger execution
        """
        self.redis_cache = redis_cache
        self.max_concurrent_executions = max_concurrent_executions
        self.execution_timeout = execution_timeout
        
        # Trigger storage
        self.triggers: Dict[str, TriggerRule] = {}
        self.triggers_by_type: Dict[TriggerType, List[str]] = defaultdict(list)
        
        # Action handlers
        self.action_handlers: Dict[ActionType, Callable] = {}
        
        # Execution tracking
        self.execution_semaphore = asyncio.Semaphore(max_concurrent_executions)
        self.execution_queue: deque = deque()
        self.active_executions: Set[str] = set()
        
        # Performance statistics
        self.stats = {
            'triggers_registered': 0,
            'events_processed': 0,
            'triggers_executed': 0,
            'actions_executed': 0,
            'execution_errors': 0,
            'average_execution_time': 0.0
        }
        
        # Safe evaluator for custom expressions
        self.evaluator = SafeEvaluator()
        
        # Register default action handlers
        self._register_default_handlers()
        
        logger.info(
            "Trigger engine initialized",
            max_concurrent_executions=max_concurrent_executions,
            execution_timeout=execution_timeout
        )
    
    def register_trigger(self, trigger_config: Union[Trigger, Dict[str, Any]]) -> str:
        """
        Register a new trigger.
        
        Args:
            trigger_config: Trigger configuration
            
        Returns:
            Trigger ID
        """
        # Convert to Trigger model if needed
        if isinstance(trigger_config, dict):
            trigger_config = Trigger(**trigger_config)
        
        # Create TriggerRule
        trigger_rule = TriggerRule(
            id=trigger_config.id,
            name=trigger_config.name,
            trigger_type=trigger_config.trigger_type,
            conditions=[
                TriggerCondition(**cond) for cond in trigger_config.conditions
            ],
            actions=[
                TriggerAction(**action) for action in trigger_config.actions
            ],
            enabled=trigger_config.enabled,
            condition_logic=trigger_config.condition_logic,
            max_executions=trigger_config.max_executions
        )
        
        # Store trigger
        self.triggers[trigger_rule.id] = trigger_rule
        self.triggers_by_type[trigger_rule.trigger_type].append(trigger_rule.id)
        
        # Update statistics
        self.stats['triggers_registered'] += 1
        
        logger.info(
            "Trigger registered",
            trigger_id=trigger_rule.id,
            trigger_name=trigger_rule.name,
            trigger_type=trigger_rule.trigger_type.value
        )
        
        return trigger_rule.id
    
    def unregister_trigger(self, trigger_id: str) -> bool:
        """
        Unregister a trigger.
        
        Args:
            trigger_id: ID of trigger to remove
            
        Returns:
            True if trigger was removed
        """
        trigger = self.triggers.get(trigger_id)
        if not trigger:
            return False
        
        # Remove from storage
        del self.triggers[trigger_id]
        self.triggers_by_type[trigger.trigger_type].remove(trigger_id)
        
        # Clean up empty lists
        if not self.triggers_by_type[trigger.trigger_type]:
            del self.triggers_by_type[trigger.trigger_type]
        
        logger.info(f"Trigger unregistered: {trigger_id}")
        return True
    
    def register_action_handler(self, action_type: ActionType, handler: Callable):
        """Register a custom action handler."""
        self.action_handlers[action_type] = handler
        logger.info(f"Action handler registered: {action_type.value}")
    
    async def process_event(
        self,
        trigger_type: TriggerType,
        event_data: Dict[str, Any],
        user_id: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> List[TriggerResult]:
        """
        Process an event and execute matching triggers.
        
        Args:
            trigger_type: Type of trigger event
            event_data: Event data for context
            user_id: Optional user ID
            session_id: Optional session ID
            
        Returns:
            List of trigger execution results
        """
        start_time = time.time()
        
        # Create execution context
        context = ExecutionContext(
            trigger_type=trigger_type,
            event_data=event_data,
            user_id=user_id,
            session_id=session_id
        )
        
        # Find matching triggers
        trigger_ids = self.triggers_by_type.get(trigger_type, [])
        matching_triggers = []
        
        for trigger_id in trigger_ids:
            trigger = self.triggers.get(trigger_id)
            if trigger and trigger.can_execute():
                # Check if conditions match
                context_dict = {
                    'event': event_data,
                    'user_id': user_id,
                    'session_id': session_id,
                    'timestamp': context.timestamp,
                    'trigger_type': trigger_type.value
                }
                
                if trigger.evaluate_conditions(context_dict):
                    matching_triggers.append(trigger)
        
        # Execute matching triggers
        execution_tasks = []
        for trigger in matching_triggers:
            task = self._execute_trigger(trigger, context)
            execution_tasks.append(task)
        
        # Wait for all executions to complete
        results = []
        if execution_tasks:
            execution_results = await asyncio.gather(*execution_tasks, return_exceptions=True)
            
            for result in execution_results:
                if isinstance(result, TriggerResult):
                    results.append(result)
                elif isinstance(result, Exception):
                    logger.error(f"Trigger execution failed: {result}")
        
        # Update statistics
        processing_time = time.time() - start_time
        self.stats['events_processed'] += 1
        
        logger.info(
            "Event processed",
            trigger_type=trigger_type.value,
            matching_triggers=len(matching_triggers),
            successful_executions=len([r for r in results if r.success]),
            processing_time=processing_time
        )
        
        return results
    
    async def _execute_trigger(
        self,
        trigger: TriggerRule,
        context: ExecutionContext
    ) -> TriggerResult:
        """Execute a single trigger with its actions."""
        start_time = time.time()
        
        # Acquire execution semaphore
        async with self.execution_semaphore:
            try:
                # Mark as active
                self.active_executions.add(trigger.id)
                
                # Prepare context for actions
                action_context = {
                    'trigger_id': trigger.id,
                    'trigger_name': trigger.name,
                    'event': context.event_data,
                    'user_id': context.user_id,
                    'session_id': context.session_id,
                    'timestamp': context.timestamp
                }
                
                # Execute actions
                actions_executed = 0
                actions_successful = 0
                
                for action in trigger.actions:
                    try:
                        actions_executed += 1
                        
                        # Execute with timeout
                        success = await asyncio.wait_for(
                            action.execute(action_context, self.action_handlers),
                            timeout=self.execution_timeout
                        )
                        
                        if success:
                            actions_successful += 1
                        
                    except asyncio.TimeoutError:
                        logger.warning(f"Action timeout in trigger {trigger.id}")
                    except Exception as e:
                        logger.error(f"Action execution error in trigger {trigger.id}: {e}")
                
                # Record execution
                trigger.record_execution()
                
                # Calculate execution time
                execution_time = time.time() - start_time
                
                # Update statistics
                self.stats['triggers_executed'] += 1
                self.stats['actions_executed'] += actions_executed
                
                # Update average execution time
                current_avg = self.stats['average_execution_time']
                total_executions = self.stats['triggers_executed']
                self.stats['average_execution_time'] = (
                    (current_avg * (total_executions - 1) + execution_time) / total_executions
                )
                
                # Create result
                result = TriggerResult(
                    trigger_id=trigger.id,
                    success=actions_successful > 0,
                    execution_time=execution_time,
                    actions_executed=actions_executed,
                    actions_successful=actions_successful,
                    context=action_context
                )
                
                logger.info(
                    "Trigger executed",
                    trigger_id=trigger.id,
                    actions_executed=actions_executed,
                    actions_successful=actions_successful,
                    execution_time=execution_time
                )
                
                return result
                
            except Exception as e:
                self.stats['execution_errors'] += 1
                execution_time = time.time() - start_time
                
                logger.error(f"Trigger execution failed: {e}")
                
                return TriggerResult(
                    trigger_id=trigger.id,
                    success=False,
                    execution_time=execution_time,
                    actions_executed=0,
                    actions_successful=0,
                    error_message=str(e)
                )
            
            finally:
                # Remove from active executions
                self.active_executions.discard(trigger.id)
    
    def _register_default_handlers(self):
        """Register default action handlers."""
        
        async def log_event_handler(params: Dict[str, Any]) -> bool:
            """Default handler for logging events."""
            message = params.get('message', 'Trigger executed')
            level = params.get('level', 'info')
            
            log_data = {
                'trigger_id': params.get('trigger_id'),
                'event': params.get('event', {}),
                'user_id': params.get('user_id')
            }
            
            if level == 'error':
                logger.error(message, **log_data)
            elif level == 'warning':
                logger.warning(message, **log_data)
            else:
                logger.info(message, **log_data)
            
            return True
        
        async def send_notification_handler(params: Dict[str, Any]) -> bool:
            """Default handler for sending notifications."""
            # This would integrate with notification system
            logger.info(
                "Notification triggered",
                message=params.get('message'),
                recipient=params.get('recipient'),
                trigger_id=params.get('trigger_id')
            )
            return True
        
        async def custom_function_handler(params: Dict[str, Any]) -> bool:
            """Handler for custom Python function execution."""
            try:
                expression = params.get('expression')
                if not expression:
                    return False
                
                # Safely evaluate the expression
                result = self.evaluator.evaluate(expression, params)
                
                logger.info(
                    "Custom function executed",
                    expression=expression,
                    result=result,
                    trigger_id=params.get('trigger_id')
                )
                
                return True
                
            except Exception as e:
                logger.error(f"Custom function execution failed: {e}")
                return False
        
        # Register handlers
        self.action_handlers[ActionType.LOG_EVENT] = log_event_handler
        self.action_handlers[ActionType.SEND_NOTIFICATION] = send_notification_handler
        self.action_handlers[ActionType.CUSTOM_FUNCTION] = custom_function_handler
    
    def get_trigger_stats(self) -> Dict[str, Any]:
        """Get trigger system statistics."""
        return {
            **self.stats,
            'registered_triggers': len(self.triggers),
            'enabled_triggers': len([t for t in self.triggers.values() if t.enabled]),
            'triggers_by_type': {
                trigger_type.value: len(trigger_ids)
                for trigger_type, trigger_ids in self.triggers_by_type.items()
            },
            'active_executions': len(self.active_executions),
            'registered_action_handlers': len(self.action_handlers)
        }
    
    def get_trigger_info(self, trigger_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific trigger."""
        trigger = self.triggers.get(trigger_id)
        if not trigger:
            return None
        
        return {
            'id': trigger.id,
            'name': trigger.name,
            'trigger_type': trigger.trigger_type.value,
            'enabled': trigger.enabled,
            'conditions_count': len(trigger.conditions),
            'actions_count': len(trigger.actions),
            'execution_count': trigger.execution_count,
            'last_executed': trigger.last_executed.isoformat() if trigger.last_executed else None,
            'created_at': trigger.created_at.isoformat(),
            'max_executions': trigger.max_executions,
            'condition_logic': trigger.condition_logic
        }
    
    async def test_trigger(
        self,
        trigger_id: str,
        test_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Test a trigger with sample data.
        
        Args:
            trigger_id: ID of trigger to test
            test_context: Test context data
            
        Returns:
            Test results
        """
        trigger = self.triggers.get(trigger_id)
        if not trigger:
            return {'error': 'Trigger not found'}
        
        try:
            # Test condition evaluation
            conditions_result = trigger.evaluate_conditions(test_context)
            
            # Test each condition individually
            condition_results = []
            for i, condition in enumerate(trigger.conditions):
                result = condition.evaluate(test_context)
                condition_results.append({
                    'condition_index': i,
                    'field': condition.field,
                    'operator': condition.operator.value,
                    'value': condition.value,
                    'result': result
                })
            
            return {
                'trigger_id': trigger_id,
                'overall_result': conditions_result,
                'condition_logic': trigger.condition_logic,
                'individual_conditions': condition_results,
                'would_execute': trigger.can_execute() and conditions_result
            }
            
        except Exception as e:
            return {
                'trigger_id': trigger_id,
                'error': str(e)
            }