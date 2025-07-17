"""
Prompt Evolution and Self-Training System.

Automatically improves prompts and system behavior based on performance feedback,
user interactions, and failure pattern analysis.
"""

import asyncio
import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class PromptTemplate:
    """Represents a prompt template with variables."""
    
    def __init__(self, template_id: str, template: str, variables: List[str], metadata: Dict[str, Any] = None):
        self.template_id = template_id
        self.template = template
        self.variables = variables
        self.metadata = metadata or {}
        self.performance_history = []
        self.version = 1
        
    def render(self, **kwargs) -> str:
        """Render template with provided variables."""
        rendered = self.template
        for var in self.variables:
            if var in kwargs:
                rendered = rendered.replace(f"{{{var}}}", str(kwargs[var]))
        return rendered
    
    def add_performance_record(self, success: bool, confidence: float, latency_ms: float, metadata: Dict[str, Any] = None):
        """Add performance record for this template."""
        record = {
            "timestamp": datetime.utcnow(),
            "success": success,
            "confidence": confidence,
            "latency_ms": latency_ms,
            "metadata": metadata or {}
        }
        self.performance_history.append(record)
        
        # Keep only recent records
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]
    
    def get_performance_metrics(self, days: int = 7) -> Dict[str, float]:
        """Get performance metrics for recent usage."""
        cutoff = datetime.utcnow() - timedelta(days=days)
        recent_records = [r for r in self.performance_history if r["timestamp"] > cutoff]
        
        if not recent_records:
            return {"success_rate": 0.0, "avg_confidence": 0.0, "avg_latency": 0.0, "usage_count": 0}
        
        return {
            "success_rate": sum(r["success"] for r in recent_records) / len(recent_records),
            "avg_confidence": sum(r["confidence"] for r in recent_records) / len(recent_records),
            "avg_latency": sum(r["latency_ms"] for r in recent_records) / len(recent_records),
            "usage_count": len(recent_records)
        }


class PromptImprovement:
    """Represents a proposed prompt improvement."""
    
    def __init__(self, original_template: PromptTemplate, improved_template: str, improvement_type: str, 
                 confidence: float, reasoning: str):
        self.original_template = original_template
        self.improved_template = improved_template
        self.improvement_type = improvement_type  # "clarity", "specificity", "error_reduction", etc.
        self.confidence = confidence
        self.reasoning = reasoning
        self.timestamp = datetime.utcnow()
        self.tested = False
        self.test_results = None


class FailurePattern:
    """Represents a detected failure pattern."""
    
    def __init__(self, pattern_id: str, description: str, frequency: int, 
                 contexts: List[Dict[str, Any]], suggested_fixes: List[str]):
        self.pattern_id = pattern_id
        self.description = description
        self.frequency = frequency
        self.contexts = contexts
        self.suggested_fixes = suggested_fixes
        self.first_seen = datetime.utcnow()
        self.last_seen = datetime.utcnow()


class PromptEvolutionEngine:
    """Engine for automatic prompt evolution and self-training."""
    
    def __init__(self):
        self.db_client = None
        self.performance_tracker = None
        self.ab_testing = None
        self.prompt_templates: Dict[str, PromptTemplate] = {}
        self.failure_patterns: Dict[str, FailurePattern] = {}
        self.improvement_queue: List[PromptImprovement] = []
        self._initialized = False
        
    async def initialize(self, config: Dict[str, Any]):
        """Initialize prompt evolution engine."""
        try:
            # Load existing prompt templates
            await self._load_prompt_templates()
            
            # Load failure patterns
            await self._load_failure_patterns()
            
            # Schedule regular analysis
            asyncio.create_task(self._schedule_analysis_loop())
            
            self._initialized = True
            logger.info("Prompt evolution engine initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize prompt evolution engine: {e}")
            raise
    
    async def register_prompt_template(self, template_id: str, template: str, variables: List[str], metadata: Dict[str, Any] = None):
        """Register a new prompt template for evolution."""
        try:
            prompt_template = PromptTemplate(template_id, template, variables, metadata)
            self.prompt_templates[template_id] = prompt_template
            
            # Store in database
            await self._store_prompt_template(prompt_template)
            
            logger.info(f"Registered prompt template: {template_id}")
            
        except Exception as e:
            logger.error(f"Failed to register prompt template: {e}")
            raise
    
    async def record_prompt_usage(self, template_id: str, success: bool, confidence: float, 
                                 latency_ms: float, context: Dict[str, Any] = None):
        """Record usage of a prompt template."""
        try:
            if template_id not in self.prompt_templates:
                logger.warning(f"Unknown template ID: {template_id}")
                return
            
            template = self.prompt_templates[template_id]
            template.add_performance_record(success, confidence, latency_ms, context)
            
            # Store in database for long-term analysis
            await self._store_prompt_usage(template_id, success, confidence, latency_ms, context)
            
            # Trigger immediate analysis if performance is degrading
            if not success or confidence < 0.5:
                await self._analyze_immediate_failure(template_id, context)
            
        except Exception as e:
            logger.error(f"Failed to record prompt usage: {e}")
            raise
    
    async def analyze_prompt_performance(self, days: int = 7) -> Dict[str, Any]:
        """Analyze performance of all prompt templates."""
        try:
            analysis_results = {}
            
            for template_id, template in self.prompt_templates.items():
                metrics = template.get_performance_metrics(days)
                
                # Analyze trends
                trends = await self._analyze_performance_trends(template_id, days)
                
                # Identify issues
                issues = []
                if metrics["success_rate"] < 0.8:
                    issues.append("Low success rate")
                if metrics["avg_confidence"] < 0.7:
                    issues.append("Low confidence")
                if metrics["avg_latency"] > 1000:
                    issues.append("High latency")
                
                analysis_results[template_id] = {
                    "metrics": metrics,
                    "trends": trends,
                    "issues": issues,
                    "improvement_priority": self._calculate_improvement_priority(metrics, issues)
                }
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Failed to analyze prompt performance: {e}")
            raise
    
    async def detect_failure_patterns(self, days: int = 7) -> List[FailurePattern]:
        """Detect common failure patterns across prompts."""
        try:
            # Get all failure records
            failure_records = await self._get_failure_records(days)
            
            # Group by similarity
            pattern_groups = self._group_similar_failures(failure_records)
            
            detected_patterns = []
            
            for group in pattern_groups:
                if len(group) >= 3:  # Minimum frequency for a pattern
                    pattern = await self._analyze_failure_group(group)
                    detected_patterns.append(pattern)
                    
                    # Store pattern
                    self.failure_patterns[pattern.pattern_id] = pattern
            
            return detected_patterns
            
        except Exception as e:
            logger.error(f"Failed to detect failure patterns: {e}")
            raise
    
    async def generate_prompt_improvements(self, template_id: str) -> List[PromptImprovement]:
        """Generate improvement suggestions for a prompt template."""
        try:
            template = self.prompt_templates.get(template_id)
            if not template:
                raise ValueError(f"Template {template_id} not found")
            
            improvements = []
            
            # Analyze current performance
            metrics = template.get_performance_metrics()
            failure_records = await self._get_template_failures(template_id)
            
            # Generate different types of improvements
            
            # 1. Clarity improvements
            clarity_improvements = await self._generate_clarity_improvements(template, failure_records)
            improvements.extend(clarity_improvements)
            
            # 2. Specificity improvements
            specificity_improvements = await self._generate_specificity_improvements(template, failure_records)
            improvements.extend(specificity_improvements)
            
            # 3. Error reduction improvements
            error_improvements = await self._generate_error_reduction_improvements(template, failure_records)
            improvements.extend(error_improvements)
            
            # 4. Performance improvements
            performance_improvements = await self._generate_performance_improvements(template, metrics)
            improvements.extend(performance_improvements)
            
            # Sort by confidence
            improvements.sort(key=lambda x: x.confidence, reverse=True)
            
            return improvements
            
        except Exception as e:
            logger.error(f"Failed to generate prompt improvements: {e}")
            raise
    
    async def test_prompt_improvement(self, improvement: PromptImprovement) -> Dict[str, Any]:
        """Test a prompt improvement using A/B testing."""
        try:
            # Create A/B test experiment
            experiment_id = await self.ab_testing.create_configuration_experiment(
                experiment_name=f"Prompt improvement for {improvement.original_template.template_id}",
                base_config={
                    "template": improvement.original_template.template,
                    "template_id": improvement.original_template.template_id
                },
                parameter_variations={
                    "template": [improvement.original_template.template, improvement.improved_template]
                }
            )
            
            # Start experiment
            await self.ab_testing.start_experiment(experiment_id)
            
            improvement.tested = True
            
            return {
                "experiment_id": experiment_id,
                "status": "started",
                "improvement_type": improvement.improvement_type,
                "estimated_duration_days": 7
            }
            
        except Exception as e:
            logger.error(f"Failed to test prompt improvement: {e}")
            raise
    
    async def apply_successful_improvements(self) -> List[str]:
        """Apply improvements that have been proven successful."""
        try:
            applied_improvements = []
            
            for improvement in self.improvement_queue:
                if improvement.tested and improvement.test_results:
                    results = improvement.test_results
                    
                    # Check if improvement is statistically significant and beneficial
                    if (results.get("is_statistically_significant", False) and 
                        results.get("improvement_beneficial", False)):
                        
                        # Apply the improvement
                        template = improvement.original_template
                        old_template = template.template
                        template.template = improvement.improved_template
                        template.version += 1
                        template.metadata["last_improved"] = datetime.utcnow()
                        template.metadata["improvement_reason"] = improvement.reasoning
                        
                        # Store updated template
                        await self._store_prompt_template(template)
                        
                        # Log the improvement
                        await self._log_improvement_application(improvement, old_template)
                        
                        applied_improvements.append(template.template_id)
                        
                        logger.info(f"Applied improvement to template {template.template_id}")
            
            # Clear applied improvements from queue
            self.improvement_queue = [i for i in self.improvement_queue if not (i.tested and i.test_results and i.test_results.get("improvement_beneficial", False))]
            
            return applied_improvements
            
        except Exception as e:
            logger.error(f"Failed to apply improvements: {e}")
            raise
    
    async def generate_training_insights(self) -> Dict[str, Any]:
        """Generate insights for system training and improvement."""
        try:
            insights = {
                "prompt_performance_summary": {},
                "common_failure_patterns": [],
                "improvement_opportunities": [],
                "system_health_score": 0.0,
                "recommendations": []
            }
            
            # Analyze prompt performance
            performance_analysis = await self.analyze_prompt_performance()
            insights["prompt_performance_summary"] = performance_analysis
            
            # Detect failure patterns
            failure_patterns = await self.detect_failure_patterns()
            insights["common_failure_patterns"] = [
                {
                    "pattern_id": p.pattern_id,
                    "description": p.description,
                    "frequency": p.frequency,
                    "suggested_fixes": p.suggested_fixes
                }
                for p in failure_patterns
            ]
            
            # Calculate system health score
            all_metrics = [metrics["metrics"] for metrics in performance_analysis.values()]
            if all_metrics:
                avg_success_rate = sum(m["success_rate"] for m in all_metrics) / len(all_metrics)
                avg_confidence = sum(m["avg_confidence"] for m in all_metrics) / len(all_metrics)
                insights["system_health_score"] = (avg_success_rate + avg_confidence) / 2
            
            # Generate recommendations
            recommendations = []
            
            if insights["system_health_score"] < 0.8:
                recommendations.append("System performance below optimal. Review prompt templates and failure patterns.")
            
            if len(failure_patterns) > 5:
                recommendations.append("Multiple failure patterns detected. Consider comprehensive prompt review.")
            
            low_performing_templates = [
                tid for tid, analysis in performance_analysis.items()
                if analysis["metrics"]["success_rate"] < 0.7
            ]
            
            if low_performing_templates:
                recommendations.append(f"Low-performing templates detected: {', '.join(low_performing_templates)}")
            
            insights["recommendations"] = recommendations
            
            return insights
            
        except Exception as e:
            logger.error(f"Failed to generate training insights: {e}")
            raise
    
    # Private helper methods
    
    async def _schedule_analysis_loop(self):
        """Schedule regular analysis of prompts and performance."""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                
                # Analyze prompt performance
                await self.analyze_prompt_performance()
                
                # Detect failure patterns
                await self.detect_failure_patterns()
                
                # Generate improvements for underperforming templates
                performance_analysis = await self.analyze_prompt_performance()
                
                for template_id, analysis in performance_analysis.items():
                    if analysis["improvement_priority"] > 0.7:
                        improvements = await self.generate_prompt_improvements(template_id)
                        
                        # Queue top improvement for testing
                        if improvements:
                            self.improvement_queue.append(improvements[0])
                
                # Apply successful improvements
                await self.apply_successful_improvements()
                
                logger.info("Completed scheduled prompt analysis")
                
            except Exception as e:
                logger.error(f"Scheduled analysis failed: {e}")
                await asyncio.sleep(300)  # Wait before retrying
    
    async def _analyze_immediate_failure(self, template_id: str, context: Dict[str, Any]):
        """Analyze immediate failure and suggest quick fixes."""
        try:
            # Quick analysis of failure context
            if context and "error_message" in context:
                error_msg = context["error_message"].lower()
                
                # Common quick fixes
                if "timeout" in error_msg:
                    # Suggest shorter prompt
                    await self._suggest_prompt_shortening(template_id)
                elif "hallucination" in error_msg or "confidence" in error_msg:
                    # Suggest more specific prompt
                    await self._suggest_prompt_specificity(template_id)
                elif "parsing" in error_msg or "format" in error_msg:
                    # Suggest clearer format instructions
                    await self._suggest_format_clarity(template_id)
            
        except Exception as e:
            logger.error(f"Immediate failure analysis failed: {e}")
    
    def _group_similar_failures(self, failure_records: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Group similar failure records together."""
        groups = []
        
        for record in failure_records:
            # Simple grouping by error type/message similarity
            placed = False
            
            for group in groups:
                if self._are_failures_similar(record, group[0]):
                    group.append(record)
                    placed = True
                    break
            
            if not placed:
                groups.append([record])
        
        return groups
    
    def _are_failures_similar(self, failure1: Dict[str, Any], failure2: Dict[str, Any]) -> bool:
        """Check if two failures are similar."""
        # Simple similarity check based on error messages and context
        msg1 = failure1.get("error_message", "").lower()
        msg2 = failure2.get("error_message", "").lower()
        
        # Check for common keywords
        common_keywords = ["timeout", "hallucination", "parsing", "format", "confidence"]
        
        for keyword in common_keywords:
            if keyword in msg1 and keyword in msg2:
                return True
        
        return False
    
    async def _analyze_failure_group(self, failure_group: List[Dict[str, Any]]) -> FailurePattern:
        """Analyze a group of similar failures to create a pattern."""
        pattern_id = f"pattern_{int(datetime.utcnow().timestamp())}"
        
        # Analyze common elements
        error_messages = [f.get("error_message", "") for f in failure_group]
        contexts = [f.get("context", {}) for f in failure_group]
        
        # Generate description
        description = self._generate_pattern_description(error_messages, contexts)
        
        # Generate suggested fixes
        suggested_fixes = self._generate_pattern_fixes(error_messages, contexts)
        
        return FailurePattern(
            pattern_id=pattern_id,
            description=description,
            frequency=len(failure_group),
            contexts=contexts,
            suggested_fixes=suggested_fixes
        )
    
    def _generate_pattern_description(self, error_messages: List[str], contexts: List[Dict[str, Any]]) -> str:
        """Generate description for a failure pattern."""
        # Analyze common themes
        common_words = self._find_common_words(error_messages)
        
        if "timeout" in common_words:
            return "Timeout errors occurring during prompt processing"
        elif "hallucination" in common_words or "confidence" in common_words:
            return "Low confidence responses with potential hallucinations"
        elif "parsing" in common_words or "format" in common_words:
            return "Response format parsing errors"
        else:
            return "General prompt processing failures"
    
    def _find_common_words(self, messages: List[str]) -> Set[str]:
        """Find common words across error messages."""
        all_words = []
        for msg in messages:
            words = re.findall(r'\w+', msg.lower())
            all_words.extend(words)
        
        # Find words that appear in at least 50% of messages
        word_counts = defaultdict(int)
        for word in all_words:
            word_counts[word] += 1
        
        threshold = len(messages) * 0.5
        return {word for word, count in word_counts.items() if count >= threshold}
    
    def _generate_pattern_fixes(self, error_messages: List[str], contexts: List[Dict[str, Any]]) -> List[str]:
        """Generate suggested fixes for a failure pattern."""
        fixes = []
        common_words = self._find_common_words(error_messages)
        
        if "timeout" in common_words:
            fixes.extend([
                "Reduce prompt length",
                "Simplify prompt structure",
                "Increase timeout threshold"
            ])
        
        if "hallucination" in common_words or "confidence" in common_words:
            fixes.extend([
                "Add more specific context",
                "Include example outputs",
                "Add confidence requirements"
            ])
        
        if "parsing" in common_words or "format" in common_words:
            fixes.extend([
                "Clarify output format requirements",
                "Add format examples",
                "Simplify expected structure"
            ])
        
        return fixes
    
    async def _generate_clarity_improvements(self, template: PromptTemplate, failure_records: List[Dict[str, Any]]) -> List[PromptImprovement]:
        """Generate clarity improvements for a template."""
        improvements = []
        
        # Analyze template for clarity issues
        current_template = template.template
        
        # Check for vague instructions
        if "please" in current_template.lower() or "try to" in current_template.lower():
            improved = current_template.replace("please ", "").replace("try to ", "")
            improvements.append(PromptImprovement(
                original_template=template,
                improved_template=improved,
                improvement_type="clarity",
                confidence=0.7,
                reasoning="Removed vague language for clearer instructions"
            ))
        
        # Check for overly complex sentences
        sentences = current_template.split('.')
        if any(len(sentence.split()) > 20 for sentence in sentences):
            # Suggest breaking down complex sentences
            improvements.append(PromptImprovement(
                original_template=template,
                improved_template=current_template,  # Would implement actual sentence simplification
                improvement_type="clarity",
                confidence=0.6,
                reasoning="Break down complex sentences for better understanding"
            ))
        
        return improvements
    
    async def _generate_specificity_improvements(self, template: PromptTemplate, failure_records: List[Dict[str, Any]]) -> List[PromptImprovement]:
        """Generate specificity improvements for a template."""
        improvements = []
        
        current_template = template.template
        
        # Check for generic terms
        generic_terms = ["thing", "stuff", "something", "anything"]
        if any(term in current_template.lower() for term in generic_terms):
            improvements.append(PromptImprovement(
                original_template=template,
                improved_template=current_template,  # Would implement specific term replacement
                improvement_type="specificity",
                confidence=0.8,
                reasoning="Replace generic terms with specific language"
            ))
        
        return improvements
    
    async def _generate_error_reduction_improvements(self, template: PromptTemplate, failure_records: List[Dict[str, Any]]) -> List[PromptImprovement]:
        """Generate error reduction improvements based on failure patterns."""
        improvements = []
        
        # Analyze failure patterns for this template
        error_messages = [f.get("error_message", "") for f in failure_records]
        
        if any("timeout" in msg.lower() for msg in error_messages):
            improvements.append(PromptImprovement(
                original_template=template,
                improved_template=template.template[:len(template.template)//2],  # Simplified shortening
                improvement_type="error_reduction",
                confidence=0.7,
                reasoning="Reduce prompt length to prevent timeouts"
            ))
        
        return improvements
    
    async def _generate_performance_improvements(self, template: PromptTemplate, metrics: Dict[str, float]) -> List[PromptImprovement]:
        """Generate performance-focused improvements."""
        improvements = []
        
        if metrics["avg_latency"] > 1000:  # High latency
            improvements.append(PromptImprovement(
                original_template=template,
                improved_template=template.template,  # Would implement latency optimizations
                improvement_type="performance",
                confidence=0.6,
                reasoning="Optimize prompt for faster processing"
            ))
        
        return improvements
    
    def _calculate_improvement_priority(self, metrics: Dict[str, float], issues: List[str]) -> float:
        """Calculate priority score for template improvement."""
        priority = 0.0
        
        # Success rate impact
        if metrics["success_rate"] < 0.8:
            priority += (0.8 - metrics["success_rate"]) * 2
        
        # Confidence impact
        if metrics["avg_confidence"] < 0.7:
            priority += (0.7 - metrics["avg_confidence"]) * 1.5
        
        # Usage impact
        if metrics["usage_count"] > 100:
            priority *= 1.5  # High usage makes improvements more impactful
        
        # Issue severity
        priority += len(issues) * 0.2
        
        return min(priority, 1.0)
    
    async def _analyze_performance_trends(self, template_id: str, days: int) -> Dict[str, str]:
        """Analyze performance trends for a template."""
        # Simplified trend analysis
        template = self.prompt_templates[template_id]
        cutoff = datetime.utcnow() - timedelta(days=days)
        recent_records = [r for r in template.performance_history if r["timestamp"] > cutoff]
        
        if len(recent_records) < 10:
            return {"success_rate": "insufficient_data", "confidence": "insufficient_data"}
        
        # Simple trend detection
        mid_point = len(recent_records) // 2
        first_half = recent_records[:mid_point]
        second_half = recent_records[mid_point:]
        
        first_success = sum(r["success"] for r in first_half) / len(first_half)
        second_success = sum(r["success"] for r in second_half) / len(second_half)
        
        success_trend = "improving" if second_success > first_success else "declining" if second_success < first_success else "stable"
        
        first_confidence = sum(r["confidence"] for r in first_half) / len(first_half)
        second_confidence = sum(r["confidence"] for r in second_half) / len(second_half)
        
        confidence_trend = "improving" if second_confidence > first_confidence else "declining" if second_confidence < first_confidence else "stable"
        
        return {"success_rate": success_trend, "confidence": confidence_trend}
    
    async def _suggest_prompt_shortening(self, template_id: str):
        """Suggest shortening a prompt to reduce timeouts."""
        logger.info(f"Quick fix suggestion: Shorten prompt {template_id} to reduce timeouts")
    
    async def _suggest_prompt_specificity(self, template_id: str):
        """Suggest making a prompt more specific."""
        logger.info(f"Quick fix suggestion: Make prompt {template_id} more specific to improve confidence")
    
    async def _suggest_format_clarity(self, template_id: str):
        """Suggest clarifying format instructions."""
        logger.info(f"Quick fix suggestion: Clarify format instructions in prompt {template_id}")
    
    # Database operations (production implementations)
    
    async def _load_prompt_templates(self):
        """Load prompt templates from database."""
        from ..utils.database import get_postgres_manager
        import json
        
        postgres = await get_postgres_manager()
        async with postgres.get_connection() as conn:
            templates = await conn.fetch(
                """
                SELECT 
                    template_id, template_name, version, template_text,
                    variables, performance_score, usage_count,
                    success_rate, avg_confidence, avg_latency_ms,
                    is_active, created_at, metadata
                FROM prompt_templates
                WHERE is_active = true
                ORDER BY template_name, version DESC
                """
            )
            
            for row in templates:
                template = PromptTemplate(
                    template_id=row['template_id'],
                    name=row['template_name'],
                    version=row['version'],
                    text=row['template_text'],
                    variables=json.loads(row['variables']) if row['variables'] else {},
                    performance_score=float(row['performance_score']) if row['performance_score'] else 0.0,
                    usage_count=row['usage_count'] or 0,
                    success_rate=float(row['success_rate']) if row['success_rate'] else 0.0,
                    average_confidence=float(row['avg_confidence']) if row['avg_confidence'] else 0.0,
                    average_latency=float(row['avg_latency_ms']) if row['avg_latency_ms'] else 0.0,
                    created_at=row['created_at'],
                    last_updated=row['created_at']
                )
                
                self.prompt_templates[row['template_id']] = template
                
            logger.info(f"Loaded {len(self.prompt_templates)} prompt templates")
    
    async def _load_failure_patterns(self):
        """Load failure patterns from database."""
        from ..utils.database import get_postgres_manager
        import json
        
        postgres = await get_postgres_manager()
        async with postgres.get_connection() as conn:
            patterns = await conn.fetch(
                """
                SELECT 
                    pattern_id, pattern_name, pattern_type,
                    detection_rules, severity, frequency,
                    suggested_fixes, confidence_threshold,
                    is_active, metadata
                FROM failure_patterns
                WHERE is_active = true
                ORDER BY severity DESC, frequency DESC
                """
            )
            
            for row in patterns:
                pattern = FailurePattern(
                    pattern_id=row['pattern_id'],
                    name=row['pattern_name'],
                    pattern_type=FailureType(row['pattern_type']),
                    indicators=json.loads(row['detection_rules']) if row['detection_rules'] else [],
                    fix_strategies=json.loads(row['suggested_fixes']) if row['suggested_fixes'] else [],
                    confidence_threshold=float(row['confidence_threshold']) if row['confidence_threshold'] else 0.8,
                    severity=row['severity'] or 'medium'
                )
                
                self.failure_patterns[row['pattern_id']] = pattern
                
            logger.info(f"Loaded {len(self.failure_patterns)} failure patterns")
    
    async def _store_prompt_template(self, template: PromptTemplate):
        """Store prompt template in database."""
        from ..utils.database import get_postgres_manager
        import json
        
        postgres = await get_postgres_manager()
        async with postgres.get_connection() as conn:
            # Deactivate previous versions
            await conn.execute(
                """
                UPDATE prompt_templates 
                SET is_active = false 
                WHERE template_name = $1 AND is_active = true
                """,
                template.name
            )
            
            # Insert new template version
            await conn.execute(
                """
                INSERT INTO prompt_templates (
                    template_id, template_name, version, template_text,
                    variables, performance_score, usage_count,
                    success_rate, avg_confidence, avg_latency_ms,
                    is_active, created_by, metadata
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13
                )
                """,
                template.template_id,
                template.name,
                template.version,
                template.text,
                json.dumps(template.variables),
                template.performance_score,
                template.usage_count,
                template.success_rate,
                template.average_confidence,
                template.average_latency,
                True,
                'prompt_evolution_engine',
                json.dumps(getattr(template, 'metadata', {}))
            )
    
    async def _store_prompt_usage(self, template_id: str, success: bool, confidence: float, latency_ms: float, context: Dict[str, Any]):
        """Store prompt usage record in database."""
        from ..utils.database import get_postgres_manager
        import json
        
        postgres = await get_postgres_manager()
        async with postgres.get_connection() as conn:
            # Record usage in history table
            await conn.execute(
                """
                INSERT INTO prompt_usage_history (
                    usage_id, template_id, session_id, user_id,
                    success, confidence_score, latency_ms,
                    input_tokens, output_tokens, cost_estimate,
                    context_data, feedback, timestamp
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13
                )
                """,
                f"usage_{template_id}_{int(datetime.utcnow().timestamp())}",
                template_id,
                context.get('session_id'),
                context.get('user_id'),
                success,
                confidence,
                latency_ms,
                context.get('input_tokens', 0),
                context.get('output_tokens', 0),
                context.get('cost_estimate', 0.0),
                json.dumps(context),
                context.get('feedback'),
                datetime.utcnow()
            )
            
            # Update aggregate statistics in templates table
            await conn.execute(
                """
                UPDATE prompt_templates 
                SET 
                    usage_count = usage_count + 1,
                    success_rate = (
                        SELECT AVG(CASE WHEN success THEN 1.0 ELSE 0.0 END)
                        FROM prompt_usage_history 
                        WHERE template_id = $1
                    ),
                    avg_confidence = (
                        SELECT AVG(confidence_score)
                        FROM prompt_usage_history 
                        WHERE template_id = $1 AND confidence_score IS NOT NULL
                    ),
                    avg_latency_ms = (
                        SELECT AVG(latency_ms)
                        FROM prompt_usage_history 
                        WHERE template_id = $1 AND latency_ms IS NOT NULL
                    ),
                    updated_at = CURRENT_TIMESTAMP
                WHERE template_id = $1
                """,
                template_id
            )
    
    async def _get_failure_records(self, days: int) -> List[Dict[str, Any]]:
        """Get failure records from database."""
        from ..utils.database import get_postgres_manager
        import json
        
        postgres = await get_postgres_manager()
        async with postgres.get_connection() as conn:
            # Get failure instances from the database
            failures = await conn.fetch(
                """
                SELECT 
                    fpi.instance_id,
                    fpi.pattern_id,
                    fpi.detected_at,
                    fpi.severity,
                    fpi.context_data,
                    fpi.impact_metrics,
                    fpi.resolution_status,
                    fp.pattern_name,
                    fp.pattern_type,
                    fp.suggested_fixes
                FROM failure_pattern_instances fpi
                JOIN failure_patterns fp ON fpi.pattern_id = fp.pattern_id
                WHERE fpi.detected_at > NOW() - INTERVAL '%s days'
                AND fpi.resolution_status != 'resolved'
                ORDER BY fpi.detected_at DESC, fpi.severity DESC
                """,
                days
            )
            
            # Get prompt-related failures specifically
            prompt_failures = await conn.fetch(
                """
                SELECT 
                    puh.template_id,
                    puh.success,
                    puh.confidence_score,
                    puh.latency_ms,
                    puh.context_data,
                    puh.feedback,
                    puh.timestamp,
                    pt.template_name
                FROM prompt_usage_history puh
                JOIN prompt_templates pt ON puh.template_id = pt.template_id
                WHERE puh.timestamp > NOW() - INTERVAL '%s days'
                AND (puh.success = false OR puh.confidence_score < 0.5)
                ORDER BY puh.timestamp DESC
                """,
                days
            )
            
            failure_records = []
            
            # Process general failures
            for failure in failures:
                failure_records.append({
                    'type': 'pattern_failure',
                    'id': failure['instance_id'],
                    'pattern_id': failure['pattern_id'],
                    'pattern_name': failure['pattern_name'],
                    'pattern_type': failure['pattern_type'],
                    'severity': failure['severity'],
                    'timestamp': failure['detected_at'],
                    'context': json.loads(failure['context_data']) if failure['context_data'] else {},
                    'impact': json.loads(failure['impact_metrics']) if failure['impact_metrics'] else {},
                    'suggested_fixes': json.loads(failure['suggested_fixes']) if failure['suggested_fixes'] else [],
                    'resolution_status': failure['resolution_status']
                })
            
            # Process prompt failures
            for failure in prompt_failures:
                failure_records.append({
                    'type': 'prompt_failure',
                    'template_id': failure['template_id'],
                    'template_name': failure['template_name'],
                    'success': failure['success'],
                    'confidence': float(failure['confidence_score']) if failure['confidence_score'] else 0.0,
                    'latency_ms': float(failure['latency_ms']) if failure['latency_ms'] else 0.0,
                    'timestamp': failure['timestamp'],
                    'context': json.loads(failure['context_data']) if failure['context_data'] else {},
                    'feedback': failure['feedback']
                })
            
            return failure_records
    
    async def _get_template_failures(self, template_id: str) -> List[Dict[str, Any]]:
        """Get failure records for a specific template."""
        from ..utils.database import get_postgres_manager
        import json
        
        postgres = await get_postgres_manager()
        async with postgres.get_connection() as conn:
            # Get recent failures for this template
            failures = await conn.fetch(
                """
                SELECT 
                    usage_id,
                    template_id,
                    session_id,
                    success,
                    confidence_score,
                    latency_ms,
                    input_tokens,
                    output_tokens,
                    context_data,
                    feedback,
                    timestamp
                FROM prompt_usage_history
                WHERE template_id = $1
                AND (success = false OR confidence_score < 0.6)
                AND timestamp > NOW() - INTERVAL '30 days'
                ORDER BY timestamp DESC
                LIMIT 100
                """,
                template_id
            )
            
            failure_records = []
            for failure in failures:
                context_data = json.loads(failure['context_data']) if failure['context_data'] else {}
                
                failure_records.append({
                    'usage_id': failure['usage_id'],
                    'template_id': failure['template_id'],
                    'session_id': failure['session_id'],
                    'success': failure['success'],
                    'confidence': float(failure['confidence_score']) if failure['confidence_score'] else 0.0,
                    'latency_ms': float(failure['latency_ms']) if failure['latency_ms'] else 0.0,
                    'input_tokens': failure['input_tokens'] or 0,
                    'output_tokens': failure['output_tokens'] or 0,
                    'timestamp': failure['timestamp'],
                    'context': context_data,
                    'feedback': failure['feedback'],
                    'error_type': context_data.get('error_type'),
                    'error_message': context_data.get('error_message')
                })
            
            return failure_records
    
    async def _log_improvement_application(self, improvement: PromptImprovement, old_template: str):
        """Log the application of an improvement."""
        from ..utils.database import get_postgres_manager
        import json
        
        postgres = await get_postgres_manager()
        async with postgres.get_connection() as conn:
            # Log improvement application
            await conn.execute(
                """
                INSERT INTO improvement_actions (
                    action_id, action_type, target_component,
                    description, parameters, priority,
                    confidence_score, risk_score, expected_improvement,
                    status, created_at, created_by
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12
                )
                """,
                f"prompt_improvement_{improvement.improvement_id}",
                'prompt_optimization',
                f"prompt_template_{improvement.template_id}",
                improvement.description,
                json.dumps({
                    'improvement_type': improvement.improvement_type.value,
                    'changes': improvement.changes,
                    'old_template': old_template,
                    'new_template': improvement.improved_template,
                    'confidence': improvement.confidence,
                    'expected_improvement': improvement.expected_improvement
                }),
                'medium',
                improvement.confidence,
                improvement.risk_score,
                improvement.expected_improvement,
                'completed',
                datetime.utcnow(),
                'prompt_evolution_engine'
            )
            
            # Record improvement results
            await conn.execute(
                """
                INSERT INTO improvement_results (
                    result_id, action_id, execution_time,
                    success, improvement_achieved, validation_score,
                    metrics_before, metrics_after, notes
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8, $9
                )
                """,
                f"result_{improvement.improvement_id}_{int(datetime.utcnow().timestamp())}",
                f"prompt_improvement_{improvement.improvement_id}",
                datetime.utcnow(),
                True,
                improvement.expected_improvement,
                improvement.confidence,
                json.dumps({}),  # Will be filled by monitoring
                json.dumps({}),  # Will be filled by monitoring
                f"Applied prompt improvement: {improvement.description}"
            )