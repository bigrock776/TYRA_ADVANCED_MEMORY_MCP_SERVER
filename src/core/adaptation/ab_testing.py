"""
A/B Testing Framework for Model and Provider Experiments.

Enables controlled experiments for testing different models, providers,
and configurations with statistical significance testing.
"""

import asyncio
import hashlib
import json
import logging
import random
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


class ExperimentStatus(Enum):
    """Experiment status levels."""
    
    DRAFT = "draft"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class ExperimentType(Enum):
    """Types of experiments."""
    
    MODEL_COMPARISON = "model_comparison"
    PROVIDER_COMPARISON = "provider_comparison"
    CONFIGURATION = "configuration"
    PROMPT_OPTIMIZATION = "prompt_optimization"
    ROUTING_STRATEGY = "routing_strategy"


@dataclass
class ExperimentVariant:
    """Single experiment variant configuration."""
    
    variant_id: str
    name: str
    description: str
    weight: float  # Traffic allocation percentage (0.0-1.0)
    configuration: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentMetric:
    """Metric definition for experiment tracking."""
    
    metric_name: str
    metric_type: str  # "latency", "accuracy", "success_rate", "custom"
    primary: bool = False  # Is this the primary metric for decision-making
    direction: str = "higher_is_better"  # "higher_is_better" or "lower_is_better"
    target_improvement: Optional[float] = None  # Minimum improvement required
    statistical_power: float = 0.8  # Required statistical power
    significance_level: float = 0.05  # Alpha for statistical tests


@dataclass
class ExperimentResult:
    """Single measurement result for an experiment."""
    
    experiment_id: str
    variant_id: str
    user_id: str
    session_id: str
    timestamp: datetime
    metrics: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentConfiguration:
    """Complete experiment configuration."""
    
    experiment_id: str
    name: str
    description: str
    experiment_type: ExperimentType
    variants: List[ExperimentVariant]
    metrics: List[ExperimentMetric]
    start_date: datetime
    end_date: Optional[datetime] = None
    sample_size_per_variant: Optional[int] = None
    status: ExperimentStatus = ExperimentStatus.DRAFT
    inclusion_criteria: Dict[str, Any] = field(default_factory=dict)
    exclusion_criteria: Dict[str, Any] = field(default_factory=dict)
    randomization_unit: str = "user"  # "user", "session", "request"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StatisticalAnalysis:
    """Statistical analysis results for an experiment."""
    
    experiment_id: str
    metric_name: str
    control_variant: str
    treatment_variant: str
    control_mean: float
    treatment_mean: float
    effect_size: float
    confidence_interval: Tuple[float, float]
    p_value: float
    statistical_power: float
    sample_size_control: int
    sample_size_treatment: int
    is_statistically_significant: bool
    practical_significance: bool
    recommendation: str


class ABTestingFramework:
    """A/B Testing framework for controlled experiments."""
    
    def __init__(self):
        self.db_client = None
        self.cache = None
        self.active_experiments: Dict[str, ExperimentConfiguration] = {}
        self.variant_assignments: Dict[str, Dict[str, str]] = {}  # user_id -> experiment_id -> variant_id
        self._initialized = False
        
    async def initialize(self, config: Dict[str, Any]):
        """Initialize A/B testing framework."""
        try:
            # Load active experiments from database
            await self._load_active_experiments()
            
            self._initialized = True
            logger.info("A/B testing framework initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize A/B testing framework: {e}")
            raise
    
    async def create_experiment(self, config: ExperimentConfiguration) -> str:
        """Create a new experiment."""
        try:
            # Validate experiment configuration
            self._validate_experiment_config(config)
            
            # Calculate required sample sizes
            await self._calculate_sample_sizes(config)
            
            # Store experiment in database
            await self._store_experiment(config)
            
            logger.info(f"Created experiment: {config.experiment_id}")
            return config.experiment_id
            
        except Exception as e:
            logger.error(f"Failed to create experiment: {e}")
            raise
    
    async def start_experiment(self, experiment_id: str) -> bool:
        """Start an experiment."""
        try:
            experiment = await self._get_experiment(experiment_id)
            if not experiment:
                raise ValueError(f"Experiment {experiment_id} not found")
            
            if experiment.status != ExperimentStatus.DRAFT:
                raise ValueError(f"Cannot start experiment in {experiment.status.value} status")
            
            # Update status to active
            experiment.status = ExperimentStatus.ACTIVE
            experiment.start_date = datetime.utcnow()
            
            # Store updated experiment
            await self._store_experiment(experiment)
            
            # Add to active experiments
            self.active_experiments[experiment_id] = experiment
            
            logger.info(f"Started experiment: {experiment_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start experiment: {e}")
            raise
    
    async def stop_experiment(self, experiment_id: str, reason: str = "Manual stop") -> bool:
        """Stop an active experiment."""
        try:
            experiment = await self._get_experiment(experiment_id)
            if not experiment:
                raise ValueError(f"Experiment {experiment_id} not found")
            
            if experiment.status != ExperimentStatus.ACTIVE:
                raise ValueError(f"Cannot stop experiment in {experiment.status.value} status")
            
            # Update status
            experiment.status = ExperimentStatus.COMPLETED
            experiment.end_date = datetime.utcnow()
            experiment.metadata["stop_reason"] = reason
            
            # Store updated experiment
            await self._store_experiment(experiment)
            
            # Remove from active experiments
            if experiment_id in self.active_experiments:
                del self.active_experiments[experiment_id]
            
            logger.info(f"Stopped experiment: {experiment_id}, reason: {reason}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop experiment: {e}")
            raise
    
    async def assign_variant(self, experiment_id: str, user_id: str, session_id: Optional[str] = None) -> Optional[str]:
        """Assign a user to an experiment variant."""
        try:
            experiment = self.active_experiments.get(experiment_id)
            if not experiment:
                return None
            
            # Check inclusion/exclusion criteria
            if not await self._check_eligibility(experiment, user_id):
                return None
            
            # Get randomization unit
            randomization_key = self._get_randomization_key(experiment, user_id, session_id)
            
            # Check if already assigned
            if experiment_id in self.variant_assignments.get(randomization_key, {}):
                return self.variant_assignments[randomization_key][experiment_id]
            
            # Assign variant based on weights
            variant_id = self._assign_variant_weighted(experiment, randomization_key)
            
            # Store assignment
            if randomization_key not in self.variant_assignments:
                self.variant_assignments[randomization_key] = {}
            self.variant_assignments[randomization_key][experiment_id] = variant_id
            
            # Persist assignment
            await self._store_variant_assignment(experiment_id, randomization_key, variant_id)
            
            return variant_id
            
        except Exception as e:
            logger.error(f"Failed to assign variant: {e}")
            return None
    
    async def record_result(self, result: ExperimentResult):
        """Record an experiment result."""
        try:
            # Validate result
            experiment = self.active_experiments.get(result.experiment_id)
            if not experiment:
                logger.warning(f"Recording result for inactive experiment: {result.experiment_id}")
                return
            
            # Store result in database
            await self._store_experiment_result(result)
            
            # Update real-time statistics if needed
            await self._update_experiment_statistics(result.experiment_id)
            
        except Exception as e:
            logger.error(f"Failed to record experiment result: {e}")
            raise
    
    async def get_variant_configuration(self, experiment_id: str, variant_id: str) -> Optional[Dict[str, Any]]:
        """Get configuration for a specific variant."""
        try:
            experiment = self.active_experiments.get(experiment_id)
            if not experiment:
                return None
            
            for variant in experiment.variants:
                if variant.variant_id == variant_id:
                    return variant.configuration
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get variant configuration: {e}")
            return None
    
    async def analyze_experiment_results(self, experiment_id: str, force_analysis: bool = False) -> List[StatisticalAnalysis]:
        """Analyze experiment results for statistical significance."""
        try:
            experiment = await self._get_experiment(experiment_id)
            if not experiment:
                raise ValueError(f"Experiment {experiment_id} not found")
            
            # Check if experiment has enough data
            if not force_analysis and not await self._has_sufficient_data(experiment_id):
                logger.info(f"Experiment {experiment_id} does not have sufficient data for analysis")
                return []
            
            # Get all results
            results = await self._get_experiment_results(experiment_id)
            
            analyses = []
            
            # Analyze each metric
            for metric in experiment.metrics:
                # Compare each variant against control (assuming first variant is control)
                control_variant = experiment.variants[0]
                
                for variant in experiment.variants[1:]:
                    analysis = await self._perform_statistical_analysis(
                        experiment_id, metric, control_variant, variant, results
                    )
                    analyses.append(analysis)
            
            return analyses
            
        except Exception as e:
            logger.error(f"Failed to analyze experiment results: {e}")
            raise
    
    async def get_experiment_status(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Get current status and metrics for an experiment."""
        try:
            experiment = await self._get_experiment(experiment_id)
            if not experiment:
                return None
            
            # Get current results summary
            results_summary = await self._get_results_summary(experiment_id)
            
            # Get statistical analysis if available
            analyses = []
            if experiment.status == ExperimentStatus.ACTIVE:
                analyses = await self.analyze_experiment_results(experiment_id)
            
            return {
                "experiment_id": experiment_id,
                "name": experiment.name,
                "status": experiment.status.value,
                "start_date": experiment.start_date,
                "end_date": experiment.end_date,
                "variants": [
                    {
                        "variant_id": v.variant_id,
                        "name": v.name,
                        "weight": v.weight,
                        "sample_size": results_summary.get("sample_sizes", {}).get(v.variant_id, 0)
                    }
                    for v in experiment.variants
                ],
                "metrics": results_summary.get("metrics", {}),
                "statistical_analyses": [asdict(a) for a in analyses],
                "estimated_completion": await self._estimate_completion_date(experiment_id)
            }
            
        except Exception as e:
            logger.error(f"Failed to get experiment status: {e}")
            return None
    
    async def get_active_experiments(self) -> List[Dict[str, Any]]:
        """Get all active experiments."""
        try:
            return [
                {
                    "experiment_id": exp_id,
                    "name": exp.name,
                    "type": exp.experiment_type.value,
                    "start_date": exp.start_date,
                    "variants_count": len(exp.variants)
                }
                for exp_id, exp in self.active_experiments.items()
            ]
            
        except Exception as e:
            logger.error(f"Failed to get active experiments: {e}")
            return []
    
    async def create_model_comparison_experiment(self, experiment_name: str, models: List[Dict[str, Any]], metrics: List[str]) -> str:
        """Create a model comparison experiment."""
        experiment_id = f"model_comp_{int(datetime.utcnow().timestamp())}"
        
        # Create variants for each model
        variants = []
        weight_per_variant = 1.0 / len(models)
        
        for i, model_config in enumerate(models):
            variant = ExperimentVariant(
                variant_id=f"model_{i}",
                name=f"Model: {model_config['name']}",
                description=f"Using model {model_config['name']} with config {model_config.get('config', {})}",
                weight=weight_per_variant,
                configuration=model_config
            )
            variants.append(variant)
        
        # Create metrics
        experiment_metrics = []
        for metric_name in metrics:
            metric = ExperimentMetric(
                metric_name=metric_name,
                metric_type="latency" if "latency" in metric_name.lower() else "accuracy",
                primary=(metric_name == metrics[0]),
                direction="lower_is_better" if "latency" in metric_name.lower() else "higher_is_better"
            )
            experiment_metrics.append(metric)
        
        # Create experiment configuration
        config = ExperimentConfiguration(
            experiment_id=experiment_id,
            name=experiment_name,
            description=f"Comparing {len(models)} models across {len(metrics)} metrics",
            experiment_type=ExperimentType.MODEL_COMPARISON,
            variants=variants,
            metrics=experiment_metrics,
            start_date=datetime.utcnow(),
            sample_size_per_variant=1000  # Default sample size
        )
        
        return await self.create_experiment(config)
    
    async def create_configuration_experiment(self, experiment_name: str, base_config: Dict[str, Any], parameter_variations: Dict[str, List[Any]]) -> str:
        """Create a configuration optimization experiment."""
        experiment_id = f"config_{int(datetime.utcnow().timestamp())}"
        
        # Generate all parameter combinations
        import itertools
        
        param_names = list(parameter_variations.keys())
        param_values = list(parameter_variations.values())
        combinations = list(itertools.product(*param_values))
        
        variants = []
        weight_per_variant = 1.0 / len(combinations)
        
        for i, combination in enumerate(combinations):
            variant_config = base_config.copy()
            
            # Apply parameter variations
            for param_name, param_value in zip(param_names, combination):
                variant_config[param_name] = param_value
            
            variant = ExperimentVariant(
                variant_id=f"config_{i}",
                name=f"Config {i}",
                description=f"Configuration with {dict(zip(param_names, combination))}",
                weight=weight_per_variant,
                configuration=variant_config
            )
            variants.append(variant)
        
        # Default metrics for configuration experiments
        experiment_metrics = [
            ExperimentMetric(
                metric_name="latency_ms",
                metric_type="latency",
                primary=True,
                direction="lower_is_better"
            ),
            ExperimentMetric(
                metric_name="success_rate",
                metric_type="accuracy",
                direction="higher_is_better"
            )
        ]
        
        config = ExperimentConfiguration(
            experiment_id=experiment_id,
            name=experiment_name,
            description=f"Testing {len(combinations)} configuration combinations",
            experiment_type=ExperimentType.CONFIGURATION,
            variants=variants,
            metrics=experiment_metrics,
            start_date=datetime.utcnow(),
            sample_size_per_variant=500
        )
        
        return await self.create_experiment(config)
    
    # Private helper methods
    
    def _validate_experiment_config(self, config: ExperimentConfiguration):
        """Validate experiment configuration."""
        if not config.variants:
            raise ValueError("Experiment must have at least one variant")
        
        if not config.metrics:
            raise ValueError("Experiment must have at least one metric")
        
        # Validate weights sum to 1.0
        total_weight = sum(v.weight for v in config.variants)
        if abs(total_weight - 1.0) > 0.001:
            raise ValueError(f"Variant weights must sum to 1.0, got {total_weight}")
        
        # Validate primary metric
        primary_metrics = [m for m in config.metrics if m.primary]
        if len(primary_metrics) != 1:
            raise ValueError("Experiment must have exactly one primary metric")
    
    async def _calculate_sample_sizes(self, config: ExperimentConfiguration):
        """Calculate required sample sizes for statistical power."""
        primary_metric = next(m for m in config.metrics if m.primary)
        
        # Use power analysis to calculate sample size
        # This is a simplified calculation - real implementation would be more sophisticated
        alpha = primary_metric.significance_level
        power = primary_metric.statistical_power
        effect_size = primary_metric.target_improvement or 0.05  # 5% default
        
        # Calculate sample size per variant (simplified formula)
        # In practice, would use more sophisticated statistical methods
        z_alpha = stats.norm.ppf(1 - alpha / 2)
        z_beta = stats.norm.ppf(power)
        
        sample_size = 2 * ((z_alpha + z_beta) / effect_size) ** 2
        
        config.sample_size_per_variant = int(sample_size)
    
    def _get_randomization_key(self, experiment: ExperimentConfiguration, user_id: str, session_id: Optional[str] = None) -> str:
        """Get randomization key based on experiment settings."""
        if experiment.randomization_unit == "user":
            return user_id
        elif experiment.randomization_unit == "session" and session_id:
            return session_id
        else:
            return user_id  # Fallback to user
    
    def _assign_variant_weighted(self, experiment: ExperimentConfiguration, randomization_key: str) -> str:
        """Assign variant based on weights using deterministic hash."""
        # Create deterministic hash for consistent assignment
        hash_input = f"{experiment.experiment_id}:{randomization_key}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        
        # Normalize to [0, 1)
        random_value = (hash_value % 1000000) / 1000000.0
        
        # Assign based on cumulative weights
        cumulative_weight = 0.0
        for variant in experiment.variants:
            cumulative_weight += variant.weight
            if random_value < cumulative_weight:
                return variant.variant_id
        
        # Fallback to last variant
        return experiment.variants[-1].variant_id
    
    async def _check_eligibility(self, experiment: ExperimentConfiguration, user_id: str) -> bool:
        """Check if user is eligible for experiment."""
        try:
            # Check inclusion criteria
            if experiment.inclusion_criteria:
                for criterion, value in experiment.inclusion_criteria.items():
                    if criterion == "user_types" and isinstance(value, list):
                        # Check if user type is in allowed list
                        user_type = await self._get_user_type(user_id)
                        if user_type not in value:
                            return False
                    elif criterion == "min_session_count":
                        # Check minimum session count
                        session_count = await self._get_user_session_count(user_id)
                        if session_count < value:
                            return False
                    elif criterion == "regions" and isinstance(value, list):
                        # Check if user region is allowed
                        user_region = await self._get_user_region(user_id)
                        if user_region not in value:
                            return False
            
            # Check exclusion criteria
            if experiment.exclusion_criteria:
                for criterion, value in experiment.exclusion_criteria.items():
                    if criterion == "excluded_user_ids" and isinstance(value, list):
                        if user_id in value:
                            return False
                    elif criterion == "bot_users" and value:
                        # Exclude bot users if specified
                        if await self._is_bot_user(user_id):
                            return False
                    elif criterion == "test_users" and value:
                        # Exclude test users if specified
                        if await self._is_test_user(user_id):
                            return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking eligibility for user {user_id}: {e}")
            return False
    
    async def _perform_statistical_analysis(self, experiment_id: str, metric: ExperimentMetric, control_variant: ExperimentVariant, treatment_variant: ExperimentVariant, results: List[ExperimentResult]) -> StatisticalAnalysis:
        """Perform statistical analysis comparing two variants."""
        # Extract metric values for each variant
        control_values = [
            r.metrics.get(metric.metric_name, 0)
            for r in results
            if r.variant_id == control_variant.variant_id and metric.metric_name in r.metrics
        ]
        
        treatment_values = [
            r.metrics.get(metric.metric_name, 0)
            for r in results
            if r.variant_id == treatment_variant.variant_id and metric.metric_name in r.metrics
        ]
        
        if len(control_values) < 10 or len(treatment_values) < 10:
            # Insufficient data
            return self._create_insufficient_data_analysis(experiment_id, metric, control_variant, treatment_variant)
        
        # Perform t-test
        control_mean = np.mean(control_values)
        treatment_mean = np.mean(treatment_values)
        
        t_stat, p_value = stats.ttest_ind(treatment_values, control_values)
        
        # Calculate effect size (Cohen's d)
        pooled_std = np.sqrt(((len(control_values) - 1) * np.var(control_values, ddof=1) + 
                             (len(treatment_values) - 1) * np.var(treatment_values, ddof=1)) / 
                            (len(control_values) + len(treatment_values) - 2))
        
        effect_size = (treatment_mean - control_mean) / pooled_std if pooled_std > 0 else 0
        
        # Calculate confidence interval
        se_diff = pooled_std * np.sqrt(1/len(control_values) + 1/len(treatment_values))
        t_critical = stats.t.ppf(1 - metric.significance_level / 2, len(control_values) + len(treatment_values) - 2)
        
        mean_diff = treatment_mean - control_mean
        ci_lower = mean_diff - t_critical * se_diff
        ci_upper = mean_diff + t_critical * se_diff
        
        # Determine significance
        is_significant = p_value < metric.significance_level
        
        # Determine practical significance
        practical_significant = False
        if metric.target_improvement:
            if metric.direction == "higher_is_better":
                practical_significant = mean_diff >= metric.target_improvement
            else:
                practical_significant = -mean_diff >= metric.target_improvement
        
        # Generate recommendation
        recommendation = self._generate_recommendation(is_significant, practical_significant, effect_size, metric.direction)
        
        return StatisticalAnalysis(
            experiment_id=experiment_id,
            metric_name=metric.metric_name,
            control_variant=control_variant.variant_id,
            treatment_variant=treatment_variant.variant_id,
            control_mean=control_mean,
            treatment_mean=treatment_mean,
            effect_size=effect_size,
            confidence_interval=(ci_lower, ci_upper),
            p_value=p_value,
            statistical_power=0.8,  # Would calculate actual power
            sample_size_control=len(control_values),
            sample_size_treatment=len(treatment_values),
            is_statistically_significant=is_significant,
            practical_significance=practical_significant,
            recommendation=recommendation
        )
    
    def _create_insufficient_data_analysis(self, experiment_id: str, metric: ExperimentMetric, control_variant: ExperimentVariant, treatment_variant: ExperimentVariant) -> StatisticalAnalysis:
        """Create analysis result for insufficient data."""
        return StatisticalAnalysis(
            experiment_id=experiment_id,
            metric_name=metric.metric_name,
            control_variant=control_variant.variant_id,
            treatment_variant=treatment_variant.variant_id,
            control_mean=0.0,
            treatment_mean=0.0,
            effect_size=0.0,
            confidence_interval=(0.0, 0.0),
            p_value=1.0,
            statistical_power=0.0,
            sample_size_control=0,
            sample_size_treatment=0,
            is_statistically_significant=False,
            practical_significance=False,
            recommendation="Insufficient data for analysis. Continue collecting data."
        )
    
    def _generate_recommendation(self, is_significant: bool, practical_significant: bool, effect_size: float, direction: str) -> str:
        """Generate recommendation based on statistical analysis."""
        if is_significant and practical_significant:
            if direction == "higher_is_better" and effect_size > 0:
                return "Strong evidence favoring treatment. Recommend implementing."
            elif direction == "lower_is_better" and effect_size < 0:
                return "Strong evidence favoring treatment. Recommend implementing."
            else:
                return "Statistically significant but favors control. Do not implement."
        elif is_significant and not practical_significant:
            return "Statistically significant but effect size too small. Consider cost/benefit."
        else:
            return "No statistically significant difference detected. Continue monitoring or stop experiment."
    
    async def _load_active_experiments(self):
        """Load active experiments from database."""
        from ..utils.database import get_postgres_manager
        
        postgres = await get_postgres_manager()
        async with postgres.get_connection() as conn:
            # Load active experiments
            experiments = await conn.fetch(
                """
                SELECT 
                    e.experiment_id, e.name, e.description, e.experiment_type,
                    e.status, e.start_date, e.end_date, e.traffic_split,
                    e.success_criteria, e.metadata
                FROM ab_experiments e
                WHERE e.status IN ('running', 'paused')
                ORDER BY e.start_date DESC
                """
            )
            
            for exp_row in experiments:
                # Load variants for this experiment
                variants = await conn.fetch(
                    """
                    SELECT variant_id, name, description, configuration, metadata
                    FROM ab_experiment_variants
                    WHERE experiment_id = $1
                    ORDER BY name
                    """,
                    exp_row['experiment_id']
                )
                
                # Load metrics for this experiment
                metrics = await conn.fetch(
                    """
                    SELECT metric_name, metric_type, target_value, target_improvement
                    FROM ab_experiment_metrics
                    WHERE experiment_id = $1
                    """,
                    exp_row['experiment_id']
                )
                
                # Create experiment configuration
                config = ExperimentConfiguration(
                    experiment_id=exp_row['experiment_id'],
                    name=exp_row['name'],
                    description=exp_row['description'],
                    variants=[
                        {
                            'id': v['variant_id'],
                            'name': v['name'],
                            'config': json.loads(v['configuration']) if v['configuration'] else {}
                        }
                        for v in variants
                    ],
                    metrics=[
                        {
                            'name': m['metric_name'],
                            'type': m['metric_type'],
                            'target': float(m['target_value']) if m['target_value'] else None
                        }
                        for m in metrics
                    ],
                    traffic_split=float(exp_row['traffic_split']) if exp_row['traffic_split'] else 1.0,
                    start_date=exp_row['start_date'],
                    end_date=exp_row['end_date']
                )
                
                self.active_experiments[exp_row['experiment_id']] = config
                
            logger.info(f"Loaded {len(self.active_experiments)} active experiments")
    
    async def _store_experiment(self, experiment: ExperimentConfiguration):
        """Store experiment configuration in database."""
        from ..utils.database import get_postgres_manager
        import json
        
        postgres = await get_postgres_manager()
        async with postgres.get_connection() as conn:
            # Store main experiment
            await conn.execute(
                """
                INSERT INTO ab_experiments (
                    experiment_id, name, description, experiment_type,
                    status, start_date, end_date, traffic_split,
                    success_criteria, created_by, metadata
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                ON CONFLICT (experiment_id) DO UPDATE SET
                    name = EXCLUDED.name,
                    description = EXCLUDED.description,
                    traffic_split = EXCLUDED.traffic_split,
                    updated_at = CURRENT_TIMESTAMP
                """,
                experiment.experiment_id,
                experiment.name,
                experiment.description,
                'optimization',
                'draft',
                experiment.start_date,
                experiment.end_date,
                experiment.traffic_split,
                json.dumps(experiment.success_criteria or {}),
                'ab_testing_framework',
                json.dumps(getattr(experiment, 'metadata', {}))
            )
            
            # Store variants
            for variant in experiment.variants:
                await conn.execute(
                    """
                    INSERT INTO ab_experiment_variants (
                        variant_id, experiment_id, name, description, configuration
                    ) VALUES ($1, $2, $3, $4, $5)
                    ON CONFLICT (variant_id) DO UPDATE SET
                        name = EXCLUDED.name,
                        configuration = EXCLUDED.configuration
                    """,
                    variant['id'],
                    experiment.experiment_id,
                    variant['name'],
                    variant.get('description', ''),
                    json.dumps(variant['config'])
                )
            
            # Store metrics
            for metric in experiment.metrics:
                await conn.execute(
                    """
                    INSERT INTO ab_experiment_metrics (
                        experiment_id, metric_name, metric_type,
                        target_value, target_improvement, is_primary
                    ) VALUES ($1, $2, $3, $4, $5, $6)
                    ON CONFLICT (experiment_id, metric_name) DO UPDATE SET
                        target_value = EXCLUDED.target_value,
                        target_improvement = EXCLUDED.target_improvement
                    """,
                    experiment.experiment_id,
                    metric['name'],
                    metric['type'],
                    metric.get('target'),
                    metric.get('improvement', 0.0),
                    metric.get('primary', False)
                )
    
    async def _get_experiment(self, experiment_id: str) -> Optional[ExperimentConfiguration]:
        """Get experiment configuration from database."""
        from ..utils.database import get_postgres_manager
        import json
        
        # Check cache first
        if experiment_id in self.active_experiments:
            return self.active_experiments[experiment_id]
        
        postgres = await get_postgres_manager()
        async with postgres.get_connection() as conn:
            # Get experiment details
            exp_row = await conn.fetchrow(
                """
                SELECT 
                    experiment_id, name, description, experiment_type,
                    status, start_date, end_date, traffic_split,
                    success_criteria, metadata
                FROM ab_experiments
                WHERE experiment_id = $1
                """,
                experiment_id
            )
            
            if not exp_row:
                return None
            
            # Get variants
            variants = await conn.fetch(
                """
                SELECT variant_id, name, description, configuration, metadata
                FROM ab_experiment_variants
                WHERE experiment_id = $1
                ORDER BY name
                """,
                experiment_id
            )
            
            # Get metrics
            metrics = await conn.fetch(
                """
                SELECT metric_name, metric_type, target_value, target_improvement, is_primary
                FROM ab_experiment_metrics
                WHERE experiment_id = $1
                """,
                experiment_id
            )
            
            # Create configuration object
            config = ExperimentConfiguration(
                experiment_id=exp_row['experiment_id'],
                name=exp_row['name'],
                description=exp_row['description'],
                variants=[
                    {
                        'id': v['variant_id'],
                        'name': v['name'],
                        'config': json.loads(v['configuration']) if v['configuration'] else {}
                    }
                    for v in variants
                ],
                metrics=[
                    {
                        'name': m['metric_name'],
                        'type': m['metric_type'],
                        'target': float(m['target_value']) if m['target_value'] else None,
                        'primary': m['is_primary']
                    }
                    for m in metrics
                ],
                traffic_split=float(exp_row['traffic_split']) if exp_row['traffic_split'] else 1.0,
                start_date=exp_row['start_date'],
                end_date=exp_row['end_date'],
                success_criteria=json.loads(exp_row['success_criteria']) if exp_row['success_criteria'] else {}
            )
            
            return config
    
    async def _store_variant_assignment(self, experiment_id: str, randomization_key: str, variant_id: str):
        """Store variant assignment in database."""
        from ..utils.database import get_postgres_manager
        
        postgres = await get_postgres_manager()
        async with postgres.get_connection() as conn:
            await conn.execute(
                """
                INSERT INTO ab_variant_assignments (
                    experiment_id, randomization_key, variant_id, assigned_at
                ) VALUES ($1, $2, $3, $4)
                ON CONFLICT (experiment_id, randomization_key) DO UPDATE SET
                    variant_id = EXCLUDED.variant_id,
                    assigned_at = EXCLUDED.assigned_at
                """,
                experiment_id,
                randomization_key,
                variant_id,
                datetime.utcnow()
            )
    
    async def _store_experiment_result(self, result: ExperimentResult):
        """Store experiment result in database."""
        from ..utils.database import get_postgres_manager
        import json
        
        postgres = await get_postgres_manager()
        async with postgres.get_connection() as conn:
            await conn.execute(
                """
                INSERT INTO ab_experiment_results (
                    experiment_id, variant_id, randomization_key,
                    session_id, metric_name, metric_value,
                    timestamp, metadata
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                """,
                result.experiment_id,
                result.variant_id,
                result.randomization_key,
                result.session_id,
                result.metric_name,
                result.metric_value,
                result.timestamp,
                json.dumps(result.metadata or {})
            )
    
    async def _update_experiment_statistics(self, experiment_id: str):
        """Update real-time experiment statistics."""
        from ..utils.database import get_postgres_manager, get_redis_manager
        import json
        
        postgres = await get_postgres_manager()
        redis = await get_redis_manager()
        
        async with postgres.get_connection() as conn:
            # Get variant statistics
            stats = await conn.fetch(
                """
                SELECT 
                    variant_id,
                    metric_name,
                    COUNT(*) as sample_size,
                    AVG(metric_value) as mean_value,
                    STDDEV(metric_value) as std_dev,
                    MIN(metric_value) as min_value,
                    MAX(metric_value) as max_value
                FROM ab_experiment_results
                WHERE experiment_id = $1
                AND timestamp > NOW() - INTERVAL '24 hours'
                GROUP BY variant_id, metric_name
                ORDER BY variant_id, metric_name
                """,
                experiment_id
            )
            
            # Cache statistics in Redis for fast access
            stats_data = {}
            for stat in stats:
                key = f"{stat['variant_id']}_{stat['metric_name']}"
                stats_data[key] = {
                    'sample_size': stat['sample_size'],
                    'mean': float(stat['mean_value']) if stat['mean_value'] else 0.0,
                    'std_dev': float(stat['std_dev']) if stat['std_dev'] else 0.0,
                    'min': float(stat['min_value']) if stat['min_value'] else 0.0,
                    'max': float(stat['max_value']) if stat['max_value'] else 0.0
                }
            
            await redis.client.setex(
                f"experiment_stats:{experiment_id}",
                300,  # Cache for 5 minutes
                json.dumps(stats_data)
            )
    
    async def _has_sufficient_data(self, experiment_id: str) -> bool:
        """Check if experiment has sufficient data for analysis."""
        from ..utils.database import get_postgres_manager
        
        postgres = await get_postgres_manager()
        async with postgres.get_connection() as conn:
            # Get sample sizes per variant
            sample_sizes = await conn.fetch(
                """
                SELECT 
                    variant_id,
                    COUNT(DISTINCT randomization_key) as unique_users,
                    COUNT(*) as total_events
                FROM ab_experiment_results
                WHERE experiment_id = $1
                GROUP BY variant_id
                """,
                experiment_id
            )
            
            if not sample_sizes:
                return False
            
            # Minimum thresholds for statistical significance
            min_users_per_variant = 100
            min_events_per_variant = 500
            
            for variant_stats in sample_sizes:
                if (variant_stats['unique_users'] < min_users_per_variant or 
                    variant_stats['total_events'] < min_events_per_variant):
                    return False
            
            # Check if we have at least 2 variants with sufficient data
            sufficient_variants = sum(
                1 for v in sample_sizes 
                if v['unique_users'] >= min_users_per_variant and 
                   v['total_events'] >= min_events_per_variant
            )
            
            return sufficient_variants >= 2
    
    async def _get_experiment_results(self, experiment_id: str) -> List[ExperimentResult]:
        """Get all results for an experiment."""
        from ..utils.database import get_postgres_manager
        import json
        
        postgres = await get_postgres_manager()
        async with postgres.get_connection() as conn:
            results = await conn.fetch(
                """
                SELECT 
                    experiment_id, variant_id, randomization_key,
                    session_id, metric_name, metric_value,
                    timestamp, metadata
                FROM ab_experiment_results
                WHERE experiment_id = $1
                ORDER BY timestamp DESC
                """,
                experiment_id
            )
            
            return [
                ExperimentResult(
                    experiment_id=row['experiment_id'],
                    variant_id=row['variant_id'],
                    randomization_key=row['randomization_key'],
                    session_id=row['session_id'],
                    metric_name=row['metric_name'],
                    metric_value=float(row['metric_value']),
                    timestamp=row['timestamp'],
                    metadata=json.loads(row['metadata']) if row['metadata'] else {}
                )
                for row in results
            ]
    
    async def _get_results_summary(self, experiment_id: str) -> Dict[str, Any]:
        """Get results summary for an experiment."""
        from ..utils.database import get_postgres_manager
        
        postgres = await get_postgres_manager()
        async with postgres.get_connection() as conn:
            # Get sample sizes by variant
            sample_sizes = await conn.fetch(
                """
                SELECT 
                    variant_id,
                    COUNT(DISTINCT randomization_key) as unique_users,
                    COUNT(*) as total_events,
                    MIN(timestamp) as first_event,
                    MAX(timestamp) as last_event
                FROM ab_experiment_results
                WHERE experiment_id = $1
                GROUP BY variant_id
                """,
                experiment_id
            )
            
            # Get metric summaries by variant
            metric_stats = await conn.fetch(
                """
                SELECT 
                    variant_id,
                    metric_name,
                    COUNT(*) as sample_size,
                    AVG(metric_value) as mean_value,
                    STDDEV(metric_value) as std_dev,
                    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY metric_value) as median,
                    MIN(metric_value) as min_value,
                    MAX(metric_value) as max_value
                FROM ab_experiment_results
                WHERE experiment_id = $1
                GROUP BY variant_id, metric_name
                ORDER BY variant_id, metric_name
                """,
                experiment_id
            )
            
            # Format response
            summary = {
                "sample_sizes": {
                    row['variant_id']: {
                        'unique_users': row['unique_users'],
                        'total_events': row['total_events'],
                        'duration_hours': (
                            (row['last_event'] - row['first_event']).total_seconds() / 3600
                            if row['last_event'] and row['first_event'] else 0
                        )
                    }
                    for row in sample_sizes
                },
                "metrics": {}
            }
            
            for stat in metric_stats:
                variant_id = stat['variant_id']
                metric_name = stat['metric_name']
                
                if variant_id not in summary['metrics']:
                    summary['metrics'][variant_id] = {}
                
                summary['metrics'][variant_id][metric_name] = {
                    'sample_size': stat['sample_size'],
                    'mean': float(stat['mean_value']) if stat['mean_value'] else 0.0,
                    'std_dev': float(stat['std_dev']) if stat['std_dev'] else 0.0,
                    'median': float(stat['median']) if stat['median'] else 0.0,
                    'min': float(stat['min_value']) if stat['min_value'] else 0.0,
                    'max': float(stat['max_value']) if stat['max_value'] else 0.0
                }
            
            return summary
    
    async def _estimate_completion_date(self, experiment_id: str) -> Optional[datetime]:
        """Estimate when experiment will have sufficient data."""
        from ..utils.database import get_postgres_manager
        
        postgres = await get_postgres_manager()
        async with postgres.get_connection() as conn:
            # Get current data collection rate
            rate_stats = await conn.fetchrow(
                """
                SELECT 
                    COUNT(DISTINCT randomization_key) as current_users,
                    COUNT(*) as current_events,
                    EXTRACT(EPOCH FROM (MAX(timestamp) - MIN(timestamp))) / 3600 as hours_elapsed
                FROM ab_experiment_results
                WHERE experiment_id = $1
                AND timestamp > NOW() - INTERVAL '24 hours'
                """,
                experiment_id
            )
            
            if not rate_stats or not rate_stats['hours_elapsed'] or rate_stats['hours_elapsed'] == 0:
                return None
            
            # Calculate rates per hour
            users_per_hour = rate_stats['current_users'] / rate_stats['hours_elapsed']
            events_per_hour = rate_stats['current_events'] / rate_stats['hours_elapsed']
            
            if users_per_hour <= 0 or events_per_hour <= 0:
                return None
            
            # Get number of variants
            variant_count = await conn.fetchval(
                """
                SELECT COUNT(DISTINCT variant_id)
                FROM ab_experiment_variants
                WHERE experiment_id = $1
                """,
                experiment_id
            )
            
            # Required totals for statistical significance
            min_users_total = 100 * variant_count  # 100 users per variant
            min_events_total = 500 * variant_count  # 500 events per variant
            
            # Calculate hours needed
            hours_for_users = max(0, (min_users_total - rate_stats['current_users']) / users_per_hour)
            hours_for_events = max(0, (min_events_total - rate_stats['current_events']) / events_per_hour)
            
            # Take the maximum (most conservative estimate)
            hours_needed = max(hours_for_users, hours_for_events)
            
            if hours_needed <= 0:
                return datetime.utcnow()  # Already sufficient data
            
            return datetime.utcnow() + timedelta(hours=hours_needed)

    async def _get_user_type(self, user_id: str) -> str:
        """Get user type for eligibility checking."""
        try:
            # This would query user database for user type
            # For now, return a default type
            return "standard"
        except Exception as e:
            logger.error(f"Error getting user type for {user_id}: {e}")
            return "unknown"

    async def _get_user_session_count(self, user_id: str) -> int:
        """Get user's session count for eligibility checking."""
        try:
            # This would query analytics database for session count
            # For now, return a default count
            return 5
        except Exception as e:
            logger.error(f"Error getting session count for {user_id}: {e}")
            return 0

    async def _get_user_region(self, user_id: str) -> str:
        """Get user's region for eligibility checking."""
        try:
            # This would query user database for region
            # For now, return a default region
            return "us-east-1"
        except Exception as e:
            logger.error(f"Error getting region for {user_id}: {e}")
            return "unknown"

    async def _is_bot_user(self, user_id: str) -> bool:
        """Check if user is a bot."""
        try:
            # This would check user metadata for bot indicators
            # For now, check if user_id starts with 'bot_'
            return user_id.startswith("bot_")
        except Exception as e:
            logger.error(f"Error checking if {user_id} is bot: {e}")
            return False

    async def _is_test_user(self, user_id: str) -> bool:
        """Check if user is a test user."""
        try:
            # This would check user metadata for test indicators
            # For now, check if user_id starts with 'test_'
            return user_id.startswith("test_")
        except Exception as e:
            logger.error(f"Error checking if {user_id} is test user: {e}")
            return False