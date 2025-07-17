"""
Event-Driven Trigger System for Tyra Memory Server.

This module provides comprehensive event-driven automation including:
- Rule-based triggers with local rule engine
- Custom user triggers using Python expressions
- Condition evaluation with sandboxed execution
- Action execution with local function calls

All processing is performed locally with zero external dependencies.
"""

from .trigger_system import (
    TriggerEngine,
    Trigger,
    TriggerRule,
    TriggerAction,
    TriggerCondition,
    TriggerType,
    ConditionOperator,
    ActionType,
    ExecutionContext,
    TriggerResult
)

__all__ = [
    "TriggerEngine",
    "Trigger",
    "TriggerRule", 
    "TriggerAction",
    "TriggerCondition",
    "TriggerType",
    "ConditionOperator",
    "ActionType",
    "ExecutionContext",
    "TriggerResult"
]