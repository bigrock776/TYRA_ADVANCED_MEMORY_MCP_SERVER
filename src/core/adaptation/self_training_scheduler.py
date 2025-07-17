"""
Self-Training Scheduler and Coordination System.

Coordinates scheduled improvement jobs, pattern detection, and automated system
optimization across all self-learning components.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from .memory_health import MemoryHealthManager
from .config_optimizer import ConfigOptimizer
from .ab_testing import ABTestingFramework
from .prompt_evolution import PromptEvolutionEngine
from ..analytics.performance_tracker import PerformanceTracker

logger = logging.getLogger(__name__)


class JobStatus(Enum):
    """Self-training job status."""
    
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobPriority(Enum):
    """Job priority levels."""
    
    CRITICAL = "critical"
    HIGH = "high" 
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class SelfTrainingJob:
    """Self-training job specification."""
    
    job_id: str
    job_type: str
    priority: JobPriority
    schedule_pattern: str  # cron-like pattern
    target_component: str
    configuration: Dict[str, Any]
    dependencies: List[str] = field(default_factory=list)
    timeout_minutes: int = 60
    retry_count: int = 3
    status: JobStatus = JobStatus.PENDING
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    results: Dict[str, Any] = field(default_factory=dict)
    error_history: List[str] = field(default_factory=list)


@dataclass
class ImprovementAction:
    """Represents a system improvement action."""
    
    action_id: str
    action_type: str
    component: str
    description: str
    confidence: float
    impact_score: float
    risk_score: float
    auto_apply: bool
    configuration_changes: Dict[str, Any]
    rollback_plan: Dict[str, Any]
    validation_checks: List[str]
    applied: bool = False
    applied_at: Optional[datetime] = None
    results: Optional[Dict[str, Any]] = None


class SelfTrainingScheduler:
    """Coordinates all self-learning and improvement activities."""
    
    def __init__(self):
        self.memory_health = None
        self.config_optimizer = None
        self.ab_testing = None
        self.prompt_evolution = None
        self.performance_tracker = None
        
        self.scheduled_jobs: Dict[str, SelfTrainingJob] = {}
        self.improvement_actions: Dict[str, ImprovementAction] = {}
        self.active_experiments: Set[str] = set()
        
        self._scheduler_running = False
        self._job_executor_pool = None
        self._initialized = False
        
    async def initialize(self, config: Dict[str, Any]):
        """Initialize self-training scheduler."""
        try:
            # Initialize component managers
            self.memory_health = MemoryHealthManager()
            await self.memory_health.initialize(config.get("memory_health", {}))
            
            self.config_optimizer = ConfigOptimizer()
            await self.config_optimizer.initialize(config.get("config_optimizer", {}))
            
            self.ab_testing = ABTestingFramework()
            await self.ab_testing.initialize(config.get("ab_testing", {}))
            
            self.prompt_evolution = PromptEvolutionEngine()
            await self.prompt_evolution.initialize(config.get("prompt_evolution", {}))
            
            self.performance_tracker = PerformanceTracker()
            
            # Create default scheduled jobs
            await self._create_default_jobs(config.get("default_jobs", {}))
            
            # Start scheduler
            await self._start_scheduler()
            
            self._initialized = True
            logger.info("Self-training scheduler initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize self-training scheduler: {e}")
            raise
    
    async def add_scheduled_job(self, job: SelfTrainingJob):
        """Add a new scheduled job."""
        try:
            # Validate job configuration
            self._validate_job(job)
            
            # Calculate next run time
            job.next_run = self._calculate_next_run(job.schedule_pattern)
            
            # Store job
            self.scheduled_jobs[job.job_id] = job
            
            # Persist to database
            await self._store_job(job)
            
            logger.info(f"Added scheduled job: {job.job_id}")
            
        except Exception as e:
            logger.error(f"Failed to add scheduled job: {e}")
            raise
    
    async def run_comprehensive_analysis(self, force: bool = False) -> Dict[str, Any]:
        """Run comprehensive system analysis and generate improvement recommendations."""
        try:
            analysis_results = {
                "timestamp": datetime.utcnow(),
                "memory_health": {},
                "performance_analysis": {},
                "prompt_analysis": {},
                "configuration_analysis": {},
                "improvement_recommendations": [],
                "risk_assessment": {},
                "overall_score": 0.0
            }
            
            # Memory health analysis
            logger.info("Running memory health analysis...")
            memory_analysis = await self.memory_health.analyze_memory_health()
            analysis_results["memory_health"] = memory_analysis
            
            # Performance analysis
            logger.info("Running performance analysis...")
            performance_analysis = await self.performance_tracker.analyze_latency_trends("all", hours=24)
            analysis_results["performance_analysis"] = performance_analysis
            
            # Prompt analysis
            logger.info("Running prompt analysis...")
            prompt_analysis = await self.prompt_evolution.analyze_prompt_performance()
            analysis_results["prompt_analysis"] = prompt_analysis
            
            # Configuration analysis
            logger.info("Running configuration analysis...")
            config_analysis = await self.config_optimizer.analyze_current_configuration()
            analysis_results["configuration_analysis"] = config_analysis
            
            # Generate improvement recommendations
            recommendations = await self._generate_improvement_recommendations(analysis_results)
            analysis_results["improvement_recommendations"] = recommendations
            
            # Calculate overall health score
            overall_score = self._calculate_overall_health_score(analysis_results)
            analysis_results["overall_score"] = overall_score
            
            # Risk assessment
            risk_assessment = await self._assess_system_risks(analysis_results)
            analysis_results["risk_assessment"] = risk_assessment
            
            # Store analysis results
            await self._store_analysis_results(analysis_results)
            
            # Trigger immediate actions if needed
            if overall_score < 0.7:
                await self._trigger_emergency_improvements(analysis_results)
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Comprehensive analysis failed: {e}")
            raise
    
    async def execute_improvement_action(self, action_id: str, dry_run: bool = False) -> Dict[str, Any]:
        """Execute a specific improvement action."""
        try:
            action = self.improvement_actions.get(action_id)
            if not action:
                raise ValueError(f"Improvement action {action_id} not found")
            
            if action.applied:
                raise ValueError(f"Action {action_id} has already been applied")
            
            execution_result = {
                "action_id": action_id,
                "dry_run": dry_run,
                "started_at": datetime.utcnow(),
                "status": "success",
                "changes_applied": [],
                "validation_results": {},
                "rollback_info": None
            }
            
            if dry_run:
                # Simulate the action
                execution_result["simulated_changes"] = action.configuration_changes
                execution_result["estimated_impact"] = action.impact_score
                execution_result["risk_factors"] = self._assess_action_risks(action)
                return execution_result
            
            try:
                # Pre-execution validation
                validation_results = await self._validate_action_preconditions(action)
                execution_result["validation_results"] = validation_results
                
                if not validation_results["all_checks_passed"]:
                    execution_result["status"] = "failed"
                    execution_result["error"] = "Pre-execution validation failed"
                    return execution_result
                
                # Create rollback checkpoint
                rollback_checkpoint = await self._create_rollback_checkpoint(action)
                execution_result["rollback_info"] = rollback_checkpoint
                
                # Execute the action based on component
                if action.component == "memory_health":
                    changes = await self._execute_memory_health_action(action)
                elif action.component == "configuration":
                    changes = await self._execute_configuration_action(action)
                elif action.component == "prompts":
                    changes = await self._execute_prompt_action(action)
                elif action.component == "performance":
                    changes = await self._execute_performance_action(action)
                else:
                    raise ValueError(f"Unknown component: {action.component}")
                
                execution_result["changes_applied"] = changes
                
                # Post-execution validation
                post_validation = await self._validate_action_results(action, changes)
                execution_result["post_validation"] = post_validation
                
                if post_validation["success"]:
                    # Mark action as applied
                    action.applied = True
                    action.applied_at = datetime.utcnow()
                    action.results = execution_result
                    
                    # Store updated action
                    await self._store_improvement_action(action)
                    
                    logger.info(f"Successfully executed improvement action: {action_id}")
                else:
                    # Rollback changes
                    await self._rollback_action(action, rollback_checkpoint)
                    execution_result["status"] = "rolled_back"
                    execution_result["rollback_reason"] = "Post-execution validation failed"
                
            except Exception as e:
                execution_result["status"] = "failed"
                execution_result["error"] = str(e)
                
                # Attempt rollback if rollback info exists
                if execution_result.get("rollback_info"):
                    try:
                        await self._rollback_action(action, execution_result["rollback_info"])
                        execution_result["rollback_completed"] = True
                    except Exception as rollback_error:
                        execution_result["rollback_error"] = str(rollback_error)
                
                logger.error(f"Failed to execute improvement action {action_id}: {e}")
            
            execution_result["completed_at"] = datetime.utcnow()
            return execution_result
            
        except Exception as e:
            logger.error(f"Failed to execute improvement action: {e}")
            raise
    
    async def start_automated_optimization_experiment(self, experiment_name: str, components: List[str]) -> str:
        """Start an automated optimization experiment."""
        try:
            # Generate experiment configurations
            experiment_configs = await self._generate_optimization_experiments(components)
            
            # Create A/B test
            experiment_id = await self.ab_testing.create_configuration_experiment(
                experiment_name=experiment_name,
                base_config=experiment_configs["baseline"],
                parameter_variations=experiment_configs["variations"]
            )
            
            # Start experiment
            await self.ab_testing.start_experiment(experiment_id)
            
            # Track experiment
            self.active_experiments.add(experiment_id)
            
            # Schedule experiment monitoring
            await self._schedule_experiment_monitoring(experiment_id)
            
            logger.info(f"Started automated optimization experiment: {experiment_id}")
            return experiment_id
            
        except Exception as e:
            logger.error(f"Failed to start optimization experiment: {e}")
            raise
    
    async def get_system_improvement_status(self) -> Dict[str, Any]:
        """Get current status of all system improvements."""
        try:
            status = {
                "overall_health_score": await self._calculate_current_health_score(),
                "active_jobs": len([j for j in self.scheduled_jobs.values() if j.status == JobStatus.RUNNING]),
                "pending_actions": len([a for a in self.improvement_actions.values() if not a.applied]),
                "active_experiments": len(self.active_experiments),
                "recent_improvements": [],
                "upcoming_jobs": [],
                "system_trends": {}
            }
            
            # Get recent improvements
            recent_improvements = [
                {
                    "action_id": a.action_id,
                    "description": a.description,
                    "applied_at": a.applied_at,
                    "impact_score": a.impact_score
                }
                for a in self.improvement_actions.values()
                if a.applied and a.applied_at and a.applied_at > datetime.utcnow() - timedelta(days=7)
            ]
            status["recent_improvements"] = sorted(recent_improvements, key=lambda x: x["applied_at"], reverse=True)
            
            # Get upcoming jobs
            upcoming_jobs = [
                {
                    "job_id": j.job_id,
                    "job_type": j.job_type,
                    "next_run": j.next_run,
                    "priority": j.priority.value
                }
                for j in self.scheduled_jobs.values()
                if j.next_run and j.next_run > datetime.utcnow()
            ]
            status["upcoming_jobs"] = sorted(upcoming_jobs, key=lambda x: x["next_run"])[:10]
            
            # Get system trends
            status["system_trends"] = await self._analyze_improvement_trends()
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to get improvement status: {e}")
            raise
    
    # Private methods
    
    async def _create_default_jobs(self, config: Dict[str, Any]):
        """Create default scheduled jobs."""
        default_jobs = [
            SelfTrainingJob(
                job_id="memory_health_check",
                job_type="memory_health_analysis",
                priority=JobPriority.HIGH,
                schedule_pattern="0 2 * * *",  # Daily at 2 AM
                target_component="memory_health",
                configuration={"comprehensive": True},
                timeout_minutes=30
            ),
            SelfTrainingJob(
                job_id="prompt_optimization",
                job_type="prompt_analysis",
                priority=JobPriority.MEDIUM,
                schedule_pattern="0 4 * * 0",  # Weekly on Sunday at 4 AM
                target_component="prompt_evolution",
                configuration={"generate_improvements": True},
                timeout_minutes=60
            ),
            SelfTrainingJob(
                job_id="configuration_optimization",
                job_type="config_analysis",
                priority=JobPriority.MEDIUM,
                schedule_pattern="0 3 1 * *",  # Monthly on 1st at 3 AM
                target_component="config_optimizer",
                configuration={"deep_analysis": True},
                timeout_minutes=90
            ),
            SelfTrainingJob(
                job_id="performance_trending",
                job_type="performance_analysis",
                priority=JobPriority.HIGH,
                schedule_pattern="0 */6 * * *",  # Every 6 hours
                target_component="performance_tracker",
                configuration={"trend_analysis": True},
                timeout_minutes=15
            ),
            SelfTrainingJob(
                job_id="comprehensive_system_review",
                job_type="comprehensive_analysis",
                priority=JobPriority.CRITICAL,
                schedule_pattern="0 1 * * 0",  # Weekly on Sunday at 1 AM
                target_component="scheduler",
                configuration={"full_analysis": True},
                timeout_minutes=120
            )
        ]
        
        for job in default_jobs:
            await self.add_scheduled_job(job)
    
    async def _start_scheduler(self):
        """Start the job scheduler."""
        if self._scheduler_running:
            return
        
        self._scheduler_running = True
        asyncio.create_task(self._scheduler_loop())
        logger.info("Self-training scheduler started")
    
    async def _scheduler_loop(self):
        """Main scheduler loop."""
        while self._scheduler_running:
            try:
                current_time = datetime.utcnow()
                
                # Check for jobs ready to run
                for job in self.scheduled_jobs.values():
                    if (job.status == JobStatus.PENDING and 
                        job.next_run and 
                        job.next_run <= current_time):
                        
                        # Execute job
                        asyncio.create_task(self._execute_job(job))
                
                # Check experiment status
                await self._check_experiment_status()
                
                # Apply ready improvements
                await self._apply_ready_improvements()
                
                # Sleep for 1 minute before next check
                await asyncio.sleep(60)
                
            except Exception as e:
                logger.error(f"Scheduler loop error: {e}")
                await asyncio.sleep(60)
    
    async def _execute_job(self, job: SelfTrainingJob):
        """Execute a scheduled job."""
        try:
            job.status = JobStatus.RUNNING
            job.last_run = datetime.utcnow()
            
            logger.info(f"Executing job: {job.job_id}")
            
            # Execute based on job type
            if job.job_type == "memory_health_analysis":
                results = await self.memory_health.analyze_memory_health()
            elif job.job_type == "prompt_analysis":
                results = await self.prompt_evolution.analyze_prompt_performance()
            elif job.job_type == "config_analysis":
                results = await self.config_optimizer.analyze_current_configuration()
            elif job.job_type == "performance_analysis":
                results = await self.performance_tracker.analyze_latency_trends("all")
            elif job.job_type == "comprehensive_analysis":
                results = await self.run_comprehensive_analysis()
            else:
                raise ValueError(f"Unknown job type: {job.job_type}")
            
            job.results = results
            job.status = JobStatus.COMPLETED
            
            # Calculate next run time
            job.next_run = self._calculate_next_run(job.schedule_pattern)
            
            logger.info(f"Job completed successfully: {job.job_id}")
            
        except Exception as e:
            job.status = JobStatus.FAILED
            job.error_history.append(f"{datetime.utcnow()}: {str(e)}")
            
            # Retry logic
            if job.retry_count > 0:
                job.retry_count -= 1
                job.status = JobStatus.PENDING
                job.next_run = datetime.utcnow() + timedelta(minutes=5)  # Retry in 5 minutes
            
            logger.error(f"Job failed: {job.job_id}: {e}")
        
        finally:
            await self._store_job(job)
    
    async def _generate_improvement_recommendations(self, analysis_results: Dict[str, Any]) -> List[ImprovementAction]:
        """Generate improvement recommendations based on analysis results."""
        recommendations = []
        
        # Memory health improvements
        memory_health = analysis_results.get("memory_health", {})
        if memory_health.get("overall_score", 1.0) < 0.8:
            action = ImprovementAction(
                action_id=f"memory_cleanup_{int(datetime.utcnow().timestamp())}",
                action_type="memory_cleanup",
                component="memory_health",
                description="Clean up stale and redundant memories",
                confidence=0.9,
                impact_score=0.7,
                risk_score=0.2,
                auto_apply=True,
                configuration_changes={"execute_cleanup": True, "aggressive": False},
                rollback_plan={"restore_from_backup": True},
                validation_checks=["verify_important_memories_preserved"]
            )
            recommendations.append(action)
        
        # Performance improvements
        performance_analysis = analysis_results.get("performance_analysis", {})
        if performance_analysis.get("trend_direction") == "declining":
            action = ImprovementAction(
                action_id=f"perf_optimization_{int(datetime.utcnow().timestamp())}",
                action_type="performance_optimization",
                component="configuration",
                description="Optimize configuration for better performance",
                confidence=0.8,
                impact_score=0.8,
                risk_score=0.3,
                auto_apply=False,  # Requires approval
                configuration_changes={"cache_ttl": 3600, "batch_size": 64},
                rollback_plan={"restore_previous_config": True},
                validation_checks=["verify_performance_improvement", "check_accuracy_maintained"]
            )
            recommendations.append(action)
        
        # Prompt improvements
        prompt_analysis = analysis_results.get("prompt_analysis", {})
        low_performing_prompts = [
            template_id for template_id, analysis in prompt_analysis.items()
            if analysis.get("metrics", {}).get("success_rate", 1.0) < 0.8
        ]
        
        if low_performing_prompts:
            action = ImprovementAction(
                action_id=f"prompt_improvement_{int(datetime.utcnow().timestamp())}",
                action_type="prompt_optimization",
                component="prompts",
                description=f"Improve {len(low_performing_prompts)} low-performing prompt templates",
                confidence=0.7,
                impact_score=0.6,
                risk_score=0.4,
                auto_apply=False,
                configuration_changes={"templates": low_performing_prompts},
                rollback_plan={"restore_original_templates": True},
                validation_checks=["verify_prompt_performance_improvement"]
            )
            recommendations.append(action)
        
        return recommendations
    
    def _calculate_overall_health_score(self, analysis_results: Dict[str, Any]) -> float:
        """Calculate overall system health score."""
        scores = []
        
        # Memory health score
        memory_score = analysis_results.get("memory_health", {}).get("overall_score", 0.5)
        scores.append(memory_score * 0.3)
        
        # Performance score (derived from trends)
        performance_analysis = analysis_results.get("performance_analysis", {})
        perf_score = 0.8 if performance_analysis.get("trend_direction") == "stable" else 0.6
        scores.append(perf_score * 0.4)
        
        # Prompt health score
        prompt_analysis = analysis_results.get("prompt_analysis", {})
        if prompt_analysis:
            prompt_scores = [
                analysis.get("metrics", {}).get("success_rate", 0.5)
                for analysis in prompt_analysis.values()
            ]
            avg_prompt_score = sum(prompt_scores) / len(prompt_scores) if prompt_scores else 0.5
            scores.append(avg_prompt_score * 0.3)
        else:
            scores.append(0.5 * 0.3)
        
        return sum(scores)
    
    async def _assess_system_risks(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess system risks based on analysis."""
        risks = {
            "high_risk_areas": [],
            "medium_risk_areas": [],
            "low_risk_areas": [],
            "overall_risk_level": "low"
        }
        
        # Check memory health risks
        memory_health = analysis_results.get("memory_health", {})
        if memory_health.get("overall_score", 1.0) < 0.6:
            risks["high_risk_areas"].append("Memory system health critically low")
        
        # Check performance risks
        performance_analysis = analysis_results.get("performance_analysis", {})
        if performance_analysis.get("trend_direction") == "declining":
            risks["medium_risk_areas"].append("Performance trending downward")
        
        # Determine overall risk level
        if risks["high_risk_areas"]:
            risks["overall_risk_level"] = "high"
        elif risks["medium_risk_areas"]:
            risks["overall_risk_level"] = "medium"
        
        return risks
    
    def _calculate_next_run(self, schedule_pattern: str) -> datetime:
        """Calculate next run time from cron-like pattern."""
        # Simplified implementation - would use proper cron parsing
        # For now, just add 24 hours for daily jobs
        if "* * *" in schedule_pattern:  # Daily pattern
            return datetime.utcnow() + timedelta(days=1)
        elif "* * 0" in schedule_pattern:  # Weekly pattern
            return datetime.utcnow() + timedelta(days=7)
        elif "1 * *" in schedule_pattern:  # Monthly pattern
            return datetime.utcnow() + timedelta(days=30)
        else:  # Default to 6 hours
            return datetime.utcnow() + timedelta(hours=6)
    
    def _validate_job(self, job: SelfTrainingJob):
        """Validate job configuration."""
        if not job.job_id:
            raise ValueError("Job ID is required")
        if not job.job_type:
            raise ValueError("Job type is required")
        if not job.target_component:
            raise ValueError("Target component is required")
    
    # Placeholder methods for database operations
    
    async def _store_job(self, job: SelfTrainingJob):
        """Store job in database."""
        from ..utils.database import get_postgres_manager
        
        postgres = await get_postgres_manager()
        async with postgres.get_connection() as conn:
            await conn.execute(
                """
                INSERT INTO analytics_jobs (
                    job_id, job_name, job_type, schedule, 
                    status, priority, parameters, created_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                ON CONFLICT (job_id) DO UPDATE SET
                    status = EXCLUDED.status,
                    updated_at = CURRENT_TIMESTAMP
                """,
                job.job_id,
                job.job_type,
                job.job_type,
                job.schedule_pattern,
                job.status.value,
                job.priority.value,
                json.dumps(job.parameters),
                datetime.utcnow()
            )
    
    async def _store_analysis_results(self, results: Dict[str, Any]):
        """Store analysis results in database."""
        from ..utils.database import get_postgres_manager
        import json
        
        postgres = await get_postgres_manager()
        async with postgres.get_connection() as conn:
            # Store in system health reports table
            await conn.execute(
                """
                INSERT INTO system_health_reports (
                    report_id, report_time, health_score,
                    component_scores, issues_detected,
                    recommendations, critical_alerts,
                    performance_summary, resource_usage,
                    error_patterns, improvement_suggestions
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11
                )
                """,
                f"analysis_{int(datetime.utcnow().timestamp())}",
                datetime.utcnow(),
                results.get('health_score', 0.0),
                json.dumps(results.get('component_scores', {})),
                results.get('issues_detected', 0),
                json.dumps(results.get('recommendations', [])),
                results.get('critical_alerts', 0),
                json.dumps(results.get('performance_summary', {})),
                json.dumps(results.get('resource_usage', {})),
                json.dumps(results.get('error_patterns', {})),
                json.dumps(results.get('improvement_suggestions', {}))
            )
    
    async def _store_improvement_action(self, action: ImprovementAction):
        """Store improvement action in database."""
        from ..utils.database import get_postgres_manager
        import json
        
        postgres = await get_postgres_manager()
        async with postgres.get_connection() as conn:
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
                ON CONFLICT (action_id) DO UPDATE SET
                    status = EXCLUDED.status,
                    updated_at = CURRENT_TIMESTAMP
                """,
                action.action_id,
                action.action_type,
                action.target_component,
                action.description,
                json.dumps(action.parameters),
                action.priority.value,
                action.confidence,
                action.risk_score,
                action.expected_improvement,
                'pending',
                datetime.utcnow(),
                'self_training_scheduler'
            )
    
    async def _trigger_emergency_improvements(self, analysis_results: Dict[str, Any]):
        """Trigger emergency improvements for critical issues."""
        critical_issues = analysis_results.get('critical_issues', [])
        
        for issue in critical_issues:
            # Create emergency improvement action
            action = ImprovementAction(
                action_id=f"emergency_{issue['component']}_{int(datetime.utcnow().timestamp())}",
                action_type="emergency_fix",
                target_component=issue['component'],
                description=f"Emergency fix for {issue['description']}",
                parameters={
                    'issue': issue,
                    'severity': 'critical',
                    'auto_apply': True
                },
                priority=JobPriority.CRITICAL,
                confidence=0.95,  # High confidence for critical issues
                risk_score=0.1,   # Low risk for emergency fixes
                expected_improvement=0.5,
                auto_apply=True
            )
            
            # Store and execute immediately
            await self._store_improvement_action(action)
            self.improvement_actions[action.action_id] = action
            
            # Execute with high priority
            try:
                await self.execute_improvement_action(action.action_id)
                logger.info(f"Emergency improvement {action.action_id} executed successfully")
            except Exception as e:
                logger.error(f"Failed to execute emergency improvement {action.action_id}: {e}")
    
    async def _validate_action_preconditions(self, action: ImprovementAction) -> Dict[str, Any]:
        """Validate preconditions for an action."""
        from ..utils.database import get_postgres_manager
        
        validation_results = {
            "all_checks_passed": True,
            "checks": {},
            "warnings": [],
            "errors": []
        }
        
        postgres = await get_postgres_manager()
        async with postgres.get_connection() as conn:
            # Check if component exists and is healthy
            health_check = await conn.fetchrow(
                """
                SELECT health_score, status 
                FROM system_health_reports 
                WHERE component_scores->$1 IS NOT NULL
                ORDER BY report_time DESC 
                LIMIT 1
                """,
                action.target_component
            )
            
            if health_check:
                component_health = float(health_check['health_score'])
                validation_results['checks']['component_health'] = component_health
                
                if component_health < 0.5:
                    validation_results['warnings'].append(
                        f"Component {action.target_component} has low health score: {component_health}"
                    )
            
            # Check for recent failures
            recent_failures = await conn.fetchval(
                """
                SELECT COUNT(*) 
                FROM improvement_results 
                WHERE action_id IN (
                    SELECT action_id FROM improvement_actions 
                    WHERE target_component = $1 
                    AND created_at > NOW() - INTERVAL '1 hour'
                )
                AND success = false
                """,
                action.target_component
            )
            
            validation_results['checks']['recent_failures'] = recent_failures
            
            if recent_failures > 2:
                validation_results['errors'].append(
                    f"Too many recent failures ({recent_failures}) for component {action.target_component}"
                )
                validation_results['all_checks_passed'] = False
            
            # Check resource availability
            resource_check = await self._check_resource_availability(action)
            validation_results['checks']['resource_available'] = resource_check
            
            if not resource_check:
                validation_results['errors'].append("Insufficient resources for action")
                validation_results['all_checks_passed'] = False
        
        return validation_results
    
    async def _create_rollback_checkpoint(self, action: ImprovementAction) -> Dict[str, Any]:
        """Create rollback checkpoint before applying action."""
        from ..utils.database import get_postgres_manager
        import json
        
        checkpoint_id = f"checkpoint_{action.action_id}_{int(datetime.utcnow().timestamp())}"
        
        postgres = await get_postgres_manager()
        async with postgres.get_connection() as conn:
            # Get current configuration for the component
            current_config = await conn.fetchrow(
                """
                SELECT parameter_path, old_value, new_value 
                FROM configuration_changes 
                WHERE component = $1 
                ORDER BY changed_at DESC 
                LIMIT 1
                """,
                action.target_component
            )
            
            checkpoint_data = {
                'component': action.target_component,
                'action_id': action.action_id,
                'timestamp': datetime.utcnow().isoformat(),
                'config_state': dict(current_config) if current_config else {},
                'parameters': action.parameters
            }
            
            # Store checkpoint in Redis for fast access
            from ..utils.database import get_redis_manager
            redis = await get_redis_manager()
            await redis.client.setex(
                f"rollback:{checkpoint_id}",
                3600,  # Keep for 1 hour
                json.dumps(checkpoint_data)
            )
            
            # Also store reference in PostgreSQL
            await conn.execute(
                """
                INSERT INTO improvement_results (
                    result_id, action_id, execution_time,
                    rollback_checkpoint
                ) VALUES ($1, $2, $3, $4)
                """,
                f"result_{action.action_id}_{int(datetime.utcnow().timestamp())}",
                action.action_id,
                datetime.utcnow(),
                checkpoint_id
            )
        
        return {
            "checkpoint_id": checkpoint_id,
            "data": checkpoint_data
        }
    
    async def _execute_memory_health_action(self, action: ImprovementAction) -> List[str]:
        """Execute memory health improvement action."""
        changes_made = []
        
        # Execute memory health improvements
        if action.parameters.get('cleanup_type') == 'cache':
            from ..cache.redis_cache import RedisCache
            cache = RedisCache()
            
            # Clear expired entries
            expired_count = await cache.clear_expired()
            changes_made.append(f"Cleared {expired_count} expired cache entries")
            
            # Optimize memory usage
            if action.parameters.get('optimize_memory', False):
                stats = await cache.optimize_memory()
                changes_made.append(f"Optimized cache memory, freed {stats['freed_bytes']} bytes")
        
        elif action.parameters.get('cleanup_type') == 'embeddings':
            from ..utils.database import get_postgres_manager
            postgres = await get_postgres_manager()
            
            async with postgres.get_connection() as conn:
                # Remove orphaned embeddings
                deleted = await conn.fetchval(
                    """
                    DELETE FROM memory_embeddings 
                    WHERE chunk_id NOT IN (SELECT chunk_id FROM memory_chunks)
                    RETURNING COUNT(*)
                    """
                )
                changes_made.append(f"Removed {deleted} orphaned embeddings")
                
                # Vacuum analyze for performance
                await conn.execute("VACUUM ANALYZE memory_embeddings")
                changes_made.append("Performed VACUUM ANALYZE on embeddings table")
        
        elif action.parameters.get('cleanup_type') == 'old_data':
            from ..utils.database import get_postgres_manager
            postgres = await get_postgres_manager()
            
            retention_days = action.parameters.get('retention_days', 90)
            async with postgres.get_connection() as conn:
                # Archive old memories
                archived = await conn.fetchval(
                    """
                    WITH archived AS (
                        UPDATE memories 
                        SET archived = true 
                        WHERE created_at < NOW() - INTERVAL '%s days'
                        AND archived = false
                        RETURNING memory_id
                    )
                    SELECT COUNT(*) FROM archived
                    """,
                    retention_days
                )
                changes_made.append(f"Archived {archived} memories older than {retention_days} days")
        
        return changes_made
    
    async def _execute_configuration_action(self, action: ImprovementAction) -> List[str]:
        """Execute configuration improvement action."""
        from ..utils.database import get_postgres_manager
        from ..utils.config import update_config
        import json
        
        changes_made = []
        postgres = await get_postgres_manager()
        
        async with postgres.get_connection() as conn:
            for param_path, new_value in action.parameters.get('config_changes', {}).items():
                # Get current value
                current_value = await conn.fetchval(
                    """
                    SELECT new_value 
                    FROM configuration_changes 
                    WHERE parameter_path = $1 
                    ORDER BY changed_at DESC 
                    LIMIT 1
                    """,
                    param_path
                )
                
                # Apply configuration change
                success = await update_config(param_path, new_value)
                
                if success:
                    # Record the change
                    await conn.execute(
                        """
                        INSERT INTO configuration_changes (
                            change_id, parameter_path, old_value, new_value,
                            component, changed_by, reason, improvement_action_id
                        ) VALUES (
                            $1, $2, $3, $4, $5, $6, $7, $8
                        )
                        """,
                        f"config_{int(datetime.utcnow().timestamp())}",
                        param_path,
                        json.dumps(current_value) if current_value else '{}',
                        json.dumps(new_value),
                        action.target_component,
                        'self_training_scheduler',
                        action.description,
                        action.action_id
                    )
                    
                    changes_made.append(f"Updated {param_path}: {current_value} -> {new_value}")
                else:
                    changes_made.append(f"Failed to update {param_path}")
        
        return changes_made
    
    async def _execute_prompt_action(self, action: ImprovementAction) -> List[str]:
        """Execute prompt improvement action."""
        from ..utils.database import get_postgres_manager
        import json
        
        changes_made = []
        postgres = await get_postgres_manager()
        
        async with postgres.get_connection() as conn:
            prompt_updates = action.parameters.get('prompt_updates', {})
            
            for template_name, updates in prompt_updates.items():
                # Get current template
                current = await conn.fetchrow(
                    """
                    SELECT template_id, version, template_text, variables
                    FROM prompt_templates
                    WHERE template_name = $1 AND is_active = true
                    """,
                    template_name
                )
                
                if current:
                    # Create new version
                    new_version = current['version'] + 1
                    new_text = updates.get('template_text', current['template_text'])
                    new_variables = updates.get('variables', json.loads(current['variables']))
                    
                    # Deactivate current version
                    await conn.execute(
                        "UPDATE prompt_templates SET is_active = false WHERE template_id = $1",
                        current['template_id']
                    )
                    
                    # Insert new version
                    await conn.execute(
                        """
                        INSERT INTO prompt_templates (
                            template_id, template_name, version, template_text,
                            variables, performance_score, usage_count,
                            is_active, created_by
                        ) VALUES (
                            $1, $2, $3, $4, $5, $6, $7, $8, $9
                        )
                        """,
                        f"{template_name}_v{new_version}_{int(datetime.utcnow().timestamp())}",
                        template_name,
                        new_version,
                        new_text,
                        json.dumps(new_variables),
                        updates.get('expected_performance', 0.0),
                        0,
                        True,
                        'self_training_scheduler'
                    )
                    
                    changes_made.append(
                        f"Updated prompt template '{template_name}' to version {new_version}"
                    )
                else:
                    changes_made.append(f"Prompt template '{template_name}' not found")
        
        return changes_made
    
    async def _execute_performance_action(self, action: ImprovementAction) -> List[str]:
        """Execute performance improvement action."""
        from ..utils.config import update_config
        
        changes_made = []
        perf_params = action.parameters.get('performance_settings', {})
        
        # Update connection pool sizes
        if 'connection_pools' in perf_params:
            for pool_name, size in perf_params['connection_pools'].items():
                success = await update_config(f"database.{pool_name}.pool_size", size)
                if success:
                    changes_made.append(f"Updated {pool_name} pool size to {size}")
        
        # Update cache settings
        if 'cache_settings' in perf_params:
            cache_config = perf_params['cache_settings']
            if 'ttl' in cache_config:
                await update_config("cache.default_ttl", cache_config['ttl'])
                changes_made.append(f"Updated cache TTL to {cache_config['ttl']}s")
            
            if 'max_size' in cache_config:
                await update_config("cache.max_size", cache_config['max_size'])
                changes_made.append(f"Updated cache max size to {cache_config['max_size']}")
        
        # Update batch processing settings
        if 'batch_settings' in perf_params:
            batch_config = perf_params['batch_settings']
            if 'embedding_batch_size' in batch_config:
                await update_config(
                    "embeddings.batch_size", 
                    batch_config['embedding_batch_size']
                )
                changes_made.append(
                    f"Updated embedding batch size to {batch_config['embedding_batch_size']}"
                )
        
        # Update query optimization settings
        if 'query_optimization' in perf_params:
            query_config = perf_params['query_optimization']
            if 'enable_parallel_queries' in query_config:
                await update_config(
                    "database.enable_parallel_queries",
                    query_config['enable_parallel_queries']
                )
                changes_made.append(
                    f"Set parallel queries to {query_config['enable_parallel_queries']}"
                )
        
        return changes_made
    
    async def _validate_action_results(self, action: ImprovementAction, changes: List[str]) -> Dict[str, Any]:
        """Validate results of an applied action."""
        from ..utils.database import get_postgres_manager
        from ..analytics.performance_tracker import get_performance_tracker
        
        validation_results = {
            "success": True,
            "changes_applied": len(changes),
            "metrics_improved": {},
            "issues": [],
            "improvement_achieved": 0.0
        }
        
        # Wait for changes to take effect
        await asyncio.sleep(2)
        
        # Get performance metrics before and after
        tracker = get_performance_tracker()
        postgres = await get_postgres_manager()
        
        async with postgres.get_connection() as conn:
            # Get baseline metrics from before the action
            baseline = await conn.fetchrow(
                """
                SELECT AVG(value) as avg_value, metric_name
                FROM performance_metrics
                WHERE metric_name IN ('latency_ms', 'throughput_rps', 'error_rate')
                AND component = $1
                AND timestamp > NOW() - INTERVAL '10 minutes'
                AND timestamp < $2
                GROUP BY metric_name
                """,
                action.target_component,
                action.execution_start_time or datetime.utcnow()
            )
            
            # Get current metrics
            current = await conn.fetchrow(
                """
                SELECT AVG(value) as avg_value, metric_name
                FROM performance_metrics
                WHERE metric_name IN ('latency_ms', 'throughput_rps', 'error_rate')
                AND component = $1
                AND timestamp > $2
                GROUP BY metric_name
                """,
                action.target_component,
                action.execution_start_time or datetime.utcnow()
            )
            
            if baseline and current:
                # Calculate improvements
                for metric in ['latency_ms', 'throughput_rps', 'error_rate']:
                    baseline_val = float(baseline['avg_value'] or 0)
                    current_val = float(current['avg_value'] or 0)
                    
                    if baseline_val > 0:
                        if metric == 'latency_ms' or metric == 'error_rate':
                            # Lower is better
                            improvement = (baseline_val - current_val) / baseline_val
                        else:
                            # Higher is better
                            improvement = (current_val - baseline_val) / baseline_val
                        
                        validation_results['metrics_improved'][metric] = improvement
                        validation_results['improvement_achieved'] += improvement / 3
            
            # Check for any errors or issues
            error_count = await conn.fetchval(
                """
                SELECT COUNT(*) 
                FROM system_logs
                WHERE level = 'ERROR'
                AND component = $1
                AND timestamp > $2
                """,
                action.target_component,
                action.execution_start_time or datetime.utcnow()
            )
            
            if error_count > 0:
                validation_results['issues'].append(f"Found {error_count} errors after applying action")
                validation_results['success'] = validation_results['improvement_achieved'] > 0
        
        # Determine overall success
        if validation_results['improvement_achieved'] < action.expected_improvement * 0.5:
            validation_results['success'] = False
            validation_results['issues'].append(
                f"Improvement {validation_results['improvement_achieved']:.2%} below expected {action.expected_improvement:.2%}"
            )
        
        return validation_results
    
    async def _rollback_action(self, action: ImprovementAction, rollback_info: Dict[str, Any]):
        """Rollback an applied action."""
        from ..utils.database import get_redis_manager, get_postgres_manager
        from ..utils.config import update_config
        import json
        
        redis = await get_redis_manager()
        postgres = await get_postgres_manager()
        
        # Get checkpoint data
        checkpoint_id = rollback_info.get('checkpoint_id')
        if not checkpoint_id:
            logger.error(f"No checkpoint ID for rollback of action {action.action_id}")
            return
        
        checkpoint_data = await redis.client.get(f"rollback:{checkpoint_id}")
        if not checkpoint_data:
            logger.error(f"Checkpoint {checkpoint_id} not found in cache")
            return
        
        checkpoint = json.loads(checkpoint_data)
        
        # Rollback based on action type
        if action.action_type == 'configuration':
            # Restore previous configuration values
            for param_path, old_value in checkpoint['config_state'].items():
                await update_config(param_path, old_value)
                logger.info(f"Rolled back {param_path} to {old_value}")
        
        elif action.action_type == 'prompt_optimization':
            # Reactivate previous prompt version
            async with postgres.get_connection() as conn:
                await conn.execute(
                    """
                    UPDATE prompt_templates 
                    SET is_active = false 
                    WHERE template_name = $1 AND is_active = true
                    """,
                    action.target_component
                )
                
                await conn.execute(
                    """
                    UPDATE prompt_templates 
                    SET is_active = true 
                    WHERE template_name = $1 
                    AND version = (
                        SELECT MAX(version) - 1 
                        FROM prompt_templates 
                        WHERE template_name = $1
                    )
                    """,
                    action.target_component
                )
        
        # Record rollback
        async with postgres.get_connection() as conn:
            await conn.execute(
                """
                UPDATE improvement_results 
                SET rollback_performed = true,
                    rollback_time = $1,
                    rollback_reason = $2
                WHERE action_id = $3
                """,
                datetime.utcnow(),
                rollback_info.get('reason', 'Action validation failed'),
                action.action_id
            )
        
        logger.info(f"Successfully rolled back action {action.action_id}")
    
    async def _check_resource_availability(self, action: ImprovementAction) -> bool:
        """Check if sufficient resources are available for the action."""
        from ..analytics.performance_tracker import get_performance_tracker
        
        tracker = get_performance_tracker()
        resource_usage = await tracker.get_resource_usage()
        
        # Check CPU usage
        if resource_usage.get('cpu_percent', 0) > 85:
            return False
        
        # Check memory usage  
        if resource_usage.get('memory_percent', 0) > 90:
            return False
        
        # Check database connections
        if action.target_component in ['database', 'postgres']:
            from ..utils.database import get_postgres_manager
            postgres = await get_postgres_manager()
            stats = await postgres.get_connection_stats()
            
            if stats.active_connections / stats.total_connections > 0.9:
                return False
        
        return True
    
    def _assess_action_risks(self, action: ImprovementAction) -> List[str]:
        """Assess risks for an action."""
        risks = []
        
        # Assess risk based on action type
        if action.action_type == 'configuration':
            # Configuration changes can affect system stability
            if 'database' in action.target_component:
                risks.append("database_connectivity_risk")
            if 'cache' in action.target_component:
                risks.append("cache_invalidation_risk")
            if any(critical in action.parameters.get('config_changes', {}) 
                   for critical in ['pool_size', 'timeout', 'max_connections']):
                risks.append("resource_exhaustion_risk")
        
        elif action.action_type == 'memory_health':
            if action.parameters.get('cleanup_type') == 'old_data':
                risks.append("data_loss_risk")
            if action.parameters.get('optimize_memory'):
                risks.append("temporary_performance_degradation_risk")
        
        elif action.action_type == 'prompt_optimization':
            risks.append("output_quality_variation_risk")
            risks.append("user_experience_change_risk")
        
        elif action.action_type == 'performance':
            if action.parameters.get('performance_settings', {}).get('enable_parallel_queries'):
                risks.append("database_load_increase_risk")
            risks.append("temporary_latency_spike_risk")
        
        # Assess based on confidence and expected improvement
        if action.confidence < 0.7:
            risks.append("low_confidence_risk")
        
        if action.expected_improvement > 0.5:
            risks.append("aggressive_optimization_risk")
        
        # Default to low risk if no specific risks identified
        if not risks:
            risks = ["low_risk"]
        
        return risks
    
    async def _generate_optimization_experiments(self, components: List[str]) -> Dict[str, Any]:
        """Generate optimization experiment configurations."""
        from ..analytics.performance_tracker import get_performance_tracker
        
        experiments = {
            "baseline": {},
            "variations": {},
            "metrics": ["latency_ms", "throughput_rps", "error_rate", "memory_usage_mb"]
        }
        
        tracker = get_performance_tracker()
        
        for component in components:
            # Get current performance baseline
            baseline_metrics = await tracker.get_component_metrics(component)
            experiments["baseline"][component] = baseline_metrics
            
            # Generate variations based on component type
            variations = []
            
            if 'embedding' in component:
                variations.extend([
                    {
                        "name": "batch_size_optimization",
                        "parameters": {"batch_size": [16, 32, 64, 128]},
                        "expected_impact": "throughput_improvement"
                    },
                    {
                        "name": "model_optimization",
                        "parameters": {"use_quantization": True},
                        "expected_impact": "memory_reduction"
                    }
                ])
            
            elif 'database' in component:
                variations.extend([
                    {
                        "name": "connection_pool_tuning",
                        "parameters": {"pool_size": [10, 20, 30, 40]},
                        "expected_impact": "concurrency_improvement"
                    },
                    {
                        "name": "query_optimization",
                        "parameters": {"enable_prepared_statements": True},
                        "expected_impact": "latency_reduction"
                    }
                ])
            
            elif 'cache' in component:
                variations.extend([
                    {
                        "name": "ttl_optimization",
                        "parameters": {"ttl_seconds": [300, 600, 1800, 3600]},
                        "expected_impact": "hit_rate_improvement"
                    },
                    {
                        "name": "eviction_policy",
                        "parameters": {"policy": ["lru", "lfu", "random"]},
                        "expected_impact": "memory_efficiency"
                    }
                ])
            
            experiments["variations"][component] = variations
        
        return experiments
    
    async def _schedule_experiment_monitoring(self, experiment_id: str):
        """Schedule monitoring for an experiment."""
        from ..utils.database import get_postgres_manager
        
        # Create monitoring job
        monitoring_job = SelfTrainingJob(
            job_id=f"monitor_experiment_{experiment_id}",
            job_type="experiment_monitoring",
            priority=JobPriority.HIGH,
            schedule_pattern="*/5 * * * *",  # Every 5 minutes
            target_component="ab_testing_framework",
            parameters={
                "experiment_id": experiment_id,
                "metrics_to_track": ["latency_ms", "throughput_rps", "error_rate"],
                "alert_thresholds": {
                    "error_rate": 0.05,
                    "latency_ms": 1000
                },
                "auto_stop_on_failure": True
            },
            status=JobStatus.PENDING
        )
        
        # Store and schedule the job
        await self._store_job(monitoring_job)
        self.scheduled_jobs[monitoring_job.job_id] = monitoring_job
        
        # Create alert rules
        postgres = await get_postgres_manager()
        async with postgres.get_connection() as conn:
            await conn.execute(
                """
                INSERT INTO performance_alerts (
                    alert_id, metric_name, component, threshold_value,
                    comparison_operator, alert_message, is_active
                ) VALUES ($1, $2, $3, $4, $5, $6, $7)
                """,
                f"alert_exp_{experiment_id}_errors",
                "error_rate",
                f"experiment_{experiment_id}",
                0.05,
                ">",
                f"Experiment {experiment_id} error rate exceeds 5%",
                True
            )
        
        logger.info(f"Scheduled monitoring for experiment {experiment_id}")
    
    async def _calculate_current_health_score(self) -> float:
        """Calculate current system health score."""
        from ..utils.database import get_postgres_manager
        from ..analytics.performance_tracker import get_performance_tracker
        
        postgres = await get_postgres_manager()
        tracker = get_performance_tracker()
        
        health_components = {
            'performance': 0.0,
            'error_rate': 0.0,
            'resource_usage': 0.0,
            'cache_effectiveness': 0.0,
            'database_health': 0.0
        }
        
        async with postgres.get_connection() as conn:
            # Performance score (based on latency)
            avg_latency = await conn.fetchval(
                """
                SELECT AVG(value) 
                FROM performance_metrics 
                WHERE metric_name = 'latency_ms' 
                AND timestamp > NOW() - INTERVAL '1 hour'
                """
            )
            
            if avg_latency:
                # Score degrades as latency increases (100ms = 1.0, 1000ms = 0.0)
                health_components['performance'] = max(0, min(1, (1000 - float(avg_latency)) / 900))
            
            # Error rate score
            error_rate = await conn.fetchval(
                """
                SELECT AVG(value) 
                FROM performance_metrics 
                WHERE metric_name = 'error_rate' 
                AND timestamp > NOW() - INTERVAL '1 hour'
                """
            )
            
            if error_rate:
                # Score degrades as error rate increases (0% = 1.0, 10% = 0.0)
                health_components['error_rate'] = max(0, min(1, 1 - (float(error_rate) * 10)))
            
            # Resource usage score
            resource_usage = await tracker.get_resource_usage()
            cpu_score = max(0, min(1, (100 - resource_usage.get('cpu_percent', 0)) / 100))
            memory_score = max(0, min(1, (100 - resource_usage.get('memory_percent', 0)) / 100))
            health_components['resource_usage'] = (cpu_score + memory_score) / 2
            
            # Cache effectiveness
            cache_stats = await conn.fetchrow(
                """
                SELECT 
                    SUM(CASE WHEN hit THEN 1 ELSE 0 END) as hits,
                    COUNT(*) as total
                FROM cache_metrics
                WHERE timestamp > NOW() - INTERVAL '1 hour'
                """
            )
            
            if cache_stats and cache_stats['total'] > 0:
                hit_rate = cache_stats['hits'] / cache_stats['total']
                health_components['cache_effectiveness'] = hit_rate
            
            # Database health (connection pool usage)
            db_health = await postgres.health_check()
            if db_health['status'] == 'healthy':
                pool_usage = db_health.get('connections_in_use', 0) / db_health.get('max_connections', 1)
                health_components['database_health'] = max(0, min(1, 1 - pool_usage))
        
        # Calculate weighted average
        weights = {
            'performance': 0.3,
            'error_rate': 0.25,
            'resource_usage': 0.2,
            'cache_effectiveness': 0.15,
            'database_health': 0.1
        }
        
        overall_score = sum(
            health_components[component] * weight 
            for component, weight in weights.items()
        )
        
        return overall_score
    
    async def _analyze_improvement_trends(self) -> Dict[str, Any]:
        """Analyze trends in system improvements."""
        from ..utils.database import get_postgres_manager
        
        postgres = await get_postgres_manager()
        trends = {
            "overall_trend": "stable",
            "component_trends": {},
            "success_rate": 0.0,
            "average_improvement": 0.0,
            "recommendations": []
        }
        
        async with postgres.get_connection() as conn:
            # Get improvement action results from last 7 days
            results = await conn.fetch(
                """
                SELECT 
                    ia.target_component,
                    ia.action_type,
                    ir.success,
                    ir.improvement_achieved,
                    ir.execution_time
                FROM improvement_actions ia
                JOIN improvement_results ir ON ia.action_id = ir.action_id
                WHERE ia.created_at > NOW() - INTERVAL '7 days'
                ORDER BY ia.created_at DESC
                """
            )
            
            if results:
                # Calculate success rate
                successful_actions = sum(1 for r in results if r['success'])
                trends['success_rate'] = successful_actions / len(results)
                
                # Calculate average improvement
                improvements = [r['improvement_achieved'] for r in results if r['improvement_achieved'] is not None]
                if improvements:
                    trends['average_improvement'] = sum(improvements) / len(improvements)
                
                # Analyze by component
                component_stats = {}
                for result in results:
                    component = result['target_component']
                    if component not in component_stats:
                        component_stats[component] = {
                            'successes': 0,
                            'failures': 0,
                            'improvements': []
                        }
                    
                    if result['success']:
                        component_stats[component]['successes'] += 1
                        if result['improvement_achieved']:
                            component_stats[component]['improvements'].append(
                                float(result['improvement_achieved'])
                            )
                    else:
                        component_stats[component]['failures'] += 1
                
                # Determine trends per component
                for component, stats in component_stats.items():
                    total_actions = stats['successes'] + stats['failures']
                    success_rate = stats['successes'] / total_actions if total_actions > 0 else 0
                    
                    avg_improvement = (
                        sum(stats['improvements']) / len(stats['improvements'])
                        if stats['improvements'] else 0
                    )
                    
                    if success_rate > 0.8 and avg_improvement > 0.1:
                        trend = "improving"
                    elif success_rate < 0.5:
                        trend = "degrading"
                    else:
                        trend = "stable"
                    
                    trends['component_trends'][component] = {
                        'trend': trend,
                        'success_rate': success_rate,
                        'average_improvement': avg_improvement
                    }
                
                # Determine overall trend
                improving_components = sum(
                    1 for t in trends['component_trends'].values() 
                    if t['trend'] == 'improving'
                )
                degrading_components = sum(
                    1 for t in trends['component_trends'].values() 
                    if t['trend'] == 'degrading'
                )
                
                if improving_components > degrading_components:
                    trends['overall_trend'] = "improving"
                elif degrading_components > improving_components:
                    trends['overall_trend'] = "degrading"
                
                # Generate recommendations
                if trends['success_rate'] < 0.7:
                    trends['recommendations'].append(
                        "Low success rate - consider more conservative optimization parameters"
                    )
                
                for component, stats in trends['component_trends'].items():
                    if stats['trend'] == 'degrading':
                        trends['recommendations'].append(
                            f"Component {component} is degrading - investigate root causes"
                        )
                    elif stats['success_rate'] < 0.5:
                        trends['recommendations'].append(
                            f"Component {component} has high failure rate - reduce optimization aggressiveness"
                        )
        
        return trends
    
    async def _check_experiment_status(self):
        """Check status of active experiments."""
        if not self.ab_testing_framework:
            return
        
        for experiment_id in list(self.active_experiments):
            try:
                # Get experiment status
                status = await self.ab_testing_framework.get_experiment_status(experiment_id)
                
                if status['status'] == 'completed':
                    # Get results and determine winner
                    results = await self.ab_testing_framework.get_experiment_results(experiment_id)
                    
                    if results['winner'] and results['confidence'] > 0.95:
                        # Create improvement action based on winning variant
                        action = ImprovementAction(
                            action_id=f"apply_experiment_{experiment_id}",
                            action_type="experiment_application",
                            target_component=status['component'],
                            description=f"Apply winning variant from experiment {experiment_id}",
                            parameters={
                                'variant_config': results['winner_config'],
                                'expected_improvement': results['improvement'],
                                'experiment_id': experiment_id
                            },
                            priority=JobPriority.HIGH,
                            confidence=results['confidence'],
                            risk_score=0.2,  # Low risk for validated experiments
                            expected_improvement=results['improvement'],
                            auto_apply=True
                        )
                        
                        await self._store_improvement_action(action)
                        self.improvement_actions[action.action_id] = action
                    
                    # Remove from active experiments
                    self.active_experiments.remove(experiment_id)
                    
                elif status['status'] == 'failed':
                    logger.error(f"Experiment {experiment_id} failed: {status.get('error')}")
                    self.active_experiments.remove(experiment_id)
                    
            except Exception as e:
                logger.error(f"Error checking experiment {experiment_id}: {e}")
    
    async def _apply_ready_improvements(self):
        """Apply improvements that are ready for auto-application."""
        for action in self.improvement_actions.values():
            if (not action.applied and action.auto_apply and 
                action.risk_score < 0.3 and action.confidence > 0.8):
                try:
                    await self.execute_improvement_action(action.action_id)
                except Exception as e:
                    logger.error(f"Failed to auto-apply improvement {action.action_id}: {e}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for the self-training scheduler."""
        return {
            "status": "healthy" if self._initialized else "unhealthy",
            "scheduler_running": self._scheduler_running,
            "active_jobs": len([j for j in self.scheduled_jobs.values() if j.status == JobStatus.RUNNING]),
            "pending_actions": len([a for a in self.improvement_actions.values() if not a.applied]),
            "active_experiments": len(self.active_experiments)
        }