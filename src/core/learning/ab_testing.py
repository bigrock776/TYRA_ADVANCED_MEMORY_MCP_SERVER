"""
Local A/B Testing Infrastructure for Statistically Valid Experiments.

This module provides comprehensive A/B testing capabilities with experiment design,
statistical significance testing using local methods, automated analysis with local tests,
and experiment monitoring. All processing is performed locally with zero external dependencies.
"""

import asyncio
import uuid
import hashlib
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Set, Any, Union, Callable, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict, deque
import json
import math
import random
from pathlib import Path

# Statistical analysis imports - all local
import scipy.stats as stats
from scipy.stats import ttest_ind, mannwhitneyu, chi2_contingency, fisher_exact
from scipy.stats import shapiro, levene, normaltest
import statsmodels.api as sm
from statsmodels.stats.power import ttest_power
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.stats.contingency_tables import mcnemar

import structlog
from pydantic import BaseModel, Field, ConfigDict

from ...core.cache.redis_cache import RedisCache
from ...core.utils.config import settings

logger = structlog.get_logger(__name__)


class ExperimentType(str, Enum):
    """Types of A/B test experiments."""
    CONVERSION = "conversion"           # Binary outcome (converted/not converted)
    CONTINUOUS = "continuous"          # Continuous metric (time, score, etc.)
    COUNT = "count"                     # Count data (clicks, views, etc.)
    PROPORTION = "proportion"           # Proportion comparison
    CATEGORICAL = "categorical"         # Categorical outcomes
    MULTIVARIATE = "multivariate"      # Multiple metrics


class ExperimentStatus(str, Enum):
    """Status of experiments."""
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    STOPPED = "stopped"
    INVALID = "invalid"


class StatisticalTest(str, Enum):
    """Types of statistical tests."""
    T_TEST = "t_test"                   # Independent t-test
    WELCH_T_TEST = "welch_t_test"      # Welch's t-test (unequal variances)
    MANN_WHITNEY = "mann_whitney"       # Mann-Whitney U test (non-parametric)
    CHI_SQUARE = "chi_square"          # Chi-square test
    FISHER_EXACT = "fisher_exact"       # Fisher's exact test
    PROPORTION_Z = "proportion_z"       # Z-test for proportions
    MCNEMAR = "mcnemar"                # McNemar's test (paired)


class TrafficAllocation(str, Enum):
    """Traffic allocation strategies."""
    EQUAL = "equal"                     # Equal split between variants
    WEIGHTED = "weighted"               # Custom weights
    ADAPTIVE = "adaptive"               # Adaptive allocation based on performance
    BAYESIAN = "bayesian"              # Bayesian optimization
    THOMPSON = "thompson"              # Thompson sampling


@dataclass
class ExperimentVariant:
    """Represents an experiment variant."""
    id: str
    name: str
    description: str
    traffic_weight: float = 0.5
    is_control: bool = False
    configuration: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())


@dataclass
class ExperimentMetric:
    """Represents a metric to track in experiments."""
    name: str
    metric_type: ExperimentType
    primary: bool = False
    direction: str = "increase"  # "increase", "decrease", "any"
    min_detectable_effect: float = 0.05
    statistical_power: float = 0.8
    significance_level: float = 0.05
    
    def get_required_sample_size(self, baseline_rate: float = 0.1) -> int:
        """Calculate required sample size for this metric."""
        if self.metric_type == ExperimentType.CONVERSION:
            # For conversion rates
            effect_size = self.min_detectable_effect
            alpha = self.significance_level
            power = self.statistical_power
            
            # Use statsmodels for sample size calculation
            try:
                from statsmodels.stats.proportion import proportion_effectsize
                from statsmodels.stats.power import ttest_power
                
                es = proportion_effectsize(baseline_rate, baseline_rate + effect_size)
                n = ttest_power(es, power, alpha, alternative='two-sided')
                return max(100, int(n * 2))  # Multiply by 2 for two groups
            except Exception:
                # Fallback calculation
                return max(100, int(16 * (1 / self.min_detectable_effect) ** 2))
        
        elif self.metric_type == ExperimentType.CONTINUOUS:
            # For continuous metrics (simplified)
            effect_size = self.min_detectable_effect
            return max(100, int(16 * (1 / effect_size) ** 2))
        
        return 1000  # Default


@dataclass
class ExperimentResult:
    """Results of an A/B test experiment."""
    experiment_id: str
    variant_results: Dict[str, Dict[str, Any]]
    statistical_tests: Dict[str, Dict[str, Any]]
    confidence_intervals: Dict[str, Dict[str, Tuple[float, float]]]
    significance: Dict[str, bool]
    effect_sizes: Dict[str, float]
    recommendations: List[str]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "experiment_id": self.experiment_id,
            "variant_results": self.variant_results,
            "statistical_tests": self.statistical_tests,
            "confidence_intervals": {
                metric: {variant: [ci[0], ci[1]] for variant, ci in variants.items()}
                for metric, variants in self.confidence_intervals.items()
            },
            "significance": self.significance,
            "effect_sizes": self.effect_sizes,
            "recommendations": self.recommendations,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class ExperimentConfig:
    """Configuration for an A/B test experiment."""
    name: str
    description: str
    variants: List[ExperimentVariant]
    metrics: List[ExperimentMetric]
    traffic_allocation: TrafficAllocation = TrafficAllocation.EQUAL
    min_sample_size: Optional[int] = None
    max_duration_days: int = 30
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    segment_filters: Dict[str, Any] = field(default_factory=dict)
    randomization_unit: str = "user_id"  # user_id, session_id, etc.
    
    def __post_init__(self):
        if self.min_sample_size is None:
            # Calculate based on primary metric
            primary_metrics = [m for m in self.metrics if m.primary]
            if primary_metrics:
                self.min_sample_size = primary_metrics[0].get_required_sample_size()
            else:
                self.min_sample_size = 1000


class StatisticalAnalyzer:
    """Performs statistical analysis for A/B tests."""
    
    def __init__(self):
        pass
    
    def check_assumptions(
        self, 
        control_data: np.ndarray, 
        treatment_data: np.ndarray,
        metric_type: ExperimentType
    ) -> Dict[str, Any]:
        """Check statistical assumptions for the data."""
        assumptions = {
            "normality": {"control": False, "treatment": False},
            "equal_variance": False,
            "independence": True,  # Assumed for A/B tests
            "sufficient_sample": {"control": False, "treatment": False}
        }
        
        # Check normality
        if len(control_data) >= 8:  # Minimum for Shapiro-Wilk
            try:
                _, p_control = shapiro(control_data)
                assumptions["normality"]["control"] = p_control > 0.05
            except Exception:
                pass
        
        if len(treatment_data) >= 8:
            try:
                _, p_treatment = shapiro(treatment_data)
                assumptions["normality"]["treatment"] = p_treatment > 0.05
            except Exception:
                pass
        
        # Check equal variance (Levene's test)
        if len(control_data) > 10 and len(treatment_data) > 10:
            try:
                _, p_levene = levene(control_data, treatment_data)
                assumptions["equal_variance"] = p_levene > 0.05
            except Exception:
                pass
        
        # Check sample size
        min_sample = 30 if metric_type == ExperimentType.CONTINUOUS else 100
        assumptions["sufficient_sample"]["control"] = len(control_data) >= min_sample
        assumptions["sufficient_sample"]["treatment"] = len(treatment_data) >= min_sample
        
        return assumptions
    
    def select_statistical_test(
        self, 
        metric_type: ExperimentType, 
        assumptions: Dict[str, Any]
    ) -> StatisticalTest:
        """Select appropriate statistical test based on data type and assumptions."""
        
        if metric_type == ExperimentType.CONVERSION:
            return StatisticalTest.PROPORTION_Z
        
        elif metric_type == ExperimentType.CONTINUOUS:
            # Check assumptions for parametric vs non-parametric
            both_normal = (assumptions["normality"]["control"] and 
                          assumptions["normality"]["treatment"])
            equal_var = assumptions["equal_variance"]
            sufficient_sample = (assumptions["sufficient_sample"]["control"] and 
                               assumptions["sufficient_sample"]["treatment"])
            
            if both_normal and equal_var and sufficient_sample:
                return StatisticalTest.T_TEST
            elif both_normal and sufficient_sample:
                return StatisticalTest.WELCH_T_TEST
            else:
                return StatisticalTest.MANN_WHITNEY
        
        elif metric_type == ExperimentType.CATEGORICAL:
            return StatisticalTest.CHI_SQUARE
        
        elif metric_type == ExperimentType.COUNT:
            return StatisticalTest.MANN_WHITNEY
        
        return StatisticalTest.T_TEST  # Default
    
    def perform_statistical_test(
        self,
        control_data: np.ndarray,
        treatment_data: np.ndarray,
        test_type: StatisticalTest,
        alternative: str = "two-sided"
    ) -> Dict[str, Any]:
        """Perform the specified statistical test."""
        
        try:
            if test_type == StatisticalTest.T_TEST:
                statistic, p_value = ttest_ind(control_data, treatment_data, 
                                             equal_var=True, alternative=alternative)
                
            elif test_type == StatisticalTest.WELCH_T_TEST:
                statistic, p_value = ttest_ind(control_data, treatment_data, 
                                             equal_var=False, alternative=alternative)
                
            elif test_type == StatisticalTest.MANN_WHITNEY:
                statistic, p_value = mannwhitneyu(control_data, treatment_data, 
                                                alternative=alternative)
                
            elif test_type == StatisticalTest.PROPORTION_Z:
                # For proportions, expect data as [successes, trials]
                control_successes = int(np.sum(control_data))
                treatment_successes = int(np.sum(treatment_data))
                control_trials = len(control_data)
                treatment_trials = len(treatment_data)
                
                counts = np.array([control_successes, treatment_successes])
                nobs = np.array([control_trials, treatment_trials])
                
                statistic, p_value = proportions_ztest(counts, nobs)
                
            elif test_type == StatisticalTest.CHI_SQUARE:
                # Expect contingency table format
                contingency_table = np.array([control_data, treatment_data])
                statistic, p_value, dof, expected = chi2_contingency(contingency_table)
                
            else:
                # Fallback to t-test
                statistic, p_value = ttest_ind(control_data, treatment_data)
            
            return {
                "test_type": test_type.value,
                "statistic": float(statistic),
                "p_value": float(p_value),
                "significant": p_value < 0.05,
                "alternative": alternative
            }
            
        except Exception as e:
            logger.error("Statistical test failed", test_type=test_type.value, error=str(e))
            return {
                "test_type": test_type.value,
                "statistic": 0.0,
                "p_value": 1.0,
                "significant": False,
                "error": str(e)
            }
    
    def calculate_confidence_interval(
        self,
        data: np.ndarray,
        confidence_level: float = 0.95,
        metric_type: ExperimentType = ExperimentType.CONTINUOUS
    ) -> Tuple[float, float]:
        """Calculate confidence interval for the metric."""
        
        try:
            if metric_type == ExperimentType.CONVERSION:
                # For proportions
                success_rate = np.mean(data)
                n = len(data)
                
                # Wilson score interval
                z = stats.norm.ppf((1 + confidence_level) / 2)
                denominator = 1 + z**2 / n
                centre_adjusted_probability = success_rate + z**2 / (2 * n)
                adjusted_standard_deviation = math.sqrt((success_rate * (1 - success_rate) + z**2 / (4 * n)) / n)
                
                lower_bound = (centre_adjusted_probability - z * adjusted_standard_deviation) / denominator
                upper_bound = (centre_adjusted_probability + z * adjusted_standard_deviation) / denominator
                
                return (max(0, lower_bound), min(1, upper_bound))
                
            else:
                # For continuous metrics
                mean = np.mean(data)
                std_err = stats.sem(data)
                h = std_err * stats.t.ppf((1 + confidence_level) / 2, len(data) - 1)
                
                return (mean - h, mean + h)
                
        except Exception as e:
            logger.warning("Failed to calculate confidence interval", error=str(e))
            mean = np.mean(data)
            return (mean, mean)
    
    def calculate_effect_size(
        self,
        control_data: np.ndarray,
        treatment_data: np.ndarray,
        metric_type: ExperimentType
    ) -> float:
        """Calculate effect size (Cohen's d for continuous, relative lift for proportions)."""
        
        try:
            if metric_type == ExperimentType.CONVERSION:
                # Relative lift for conversions
                control_rate = np.mean(control_data)
                treatment_rate = np.mean(treatment_data)
                
                if control_rate == 0:
                    return float('inf') if treatment_rate > 0 else 0
                
                return (treatment_rate - control_rate) / control_rate
                
            else:
                # Cohen's d for continuous metrics
                control_mean = np.mean(control_data)
                treatment_mean = np.mean(treatment_data)
                
                # Pooled standard deviation
                control_var = np.var(control_data, ddof=1)
                treatment_var = np.var(treatment_data, ddof=1)
                pooled_std = math.sqrt(((len(control_data) - 1) * control_var + 
                                      (len(treatment_data) - 1) * treatment_var) / 
                                     (len(control_data) + len(treatment_data) - 2))
                
                if pooled_std == 0:
                    return 0
                
                return (treatment_mean - control_mean) / pooled_std
                
        except Exception as e:
            logger.warning("Failed to calculate effect size", error=str(e))
            return 0.0


class ExperimentManager:
    """Manages A/B test experiments."""
    
    def __init__(self, cache: Optional[RedisCache] = None):
        self.cache = cache
        self.experiments: Dict[str, Dict[str, Any]] = {}
        self.experiment_data: Dict[str, Dict[str, List[Any]]] = {}
        self.statistical_analyzer = StatisticalAnalyzer()
        self.active_experiments: Set[str] = set()
        
        # Traffic allocation state
        self.user_assignments: Dict[str, Dict[str, str]] = {}  # {experiment_id: {user_id: variant_id}}
        
        logger.info("A/B testing experiment manager initialized")
    
    async def create_experiment(self, config: ExperimentConfig) -> str:
        """Create a new A/B test experiment."""
        experiment_id = str(uuid.uuid4())
        
        # Validate configuration
        if len(config.variants) < 2:
            raise ValueError("Experiment must have at least 2 variants")
        
        # Ensure traffic weights sum to 1
        total_weight = sum(v.traffic_weight for v in config.variants)
        if abs(total_weight - 1.0) > 0.01:
            # Normalize weights
            for variant in config.variants:
                variant.traffic_weight /= total_weight
        
        experiment_data = {
            "id": experiment_id,
            "config": asdict(config),
            "status": ExperimentStatus.DRAFT,
            "created_at": datetime.utcnow().isoformat(),
            "started_at": None,
            "ended_at": None,
            "participant_count": 0,
            "results": None
        }
        
        self.experiments[experiment_id] = experiment_data
        self.experiment_data[experiment_id] = {
            variant.id: [] for variant in config.variants
        }
        self.user_assignments[experiment_id] = {}
        
        # Cache experiment if available
        if self.cache:
            await self.cache.set(
                f"experiment:{experiment_id}",
                json.dumps(experiment_data, default=str),
                ttl=86400 * 30  # 30 days
            )
        
        logger.info("A/B test experiment created", experiment_id=experiment_id, name=config.name)
        return experiment_id
    
    async def start_experiment(self, experiment_id: str) -> bool:
        """Start an A/B test experiment."""
        if experiment_id not in self.experiments:
            return False
        
        experiment = self.experiments[experiment_id]
        if experiment["status"] != ExperimentStatus.DRAFT:
            return False
        
        experiment["status"] = ExperimentStatus.RUNNING
        experiment["started_at"] = datetime.utcnow().isoformat()
        self.active_experiments.add(experiment_id)
        
        # Update cache
        if self.cache:
            await self.cache.set(
                f"experiment:{experiment_id}",
                json.dumps(experiment, default=str),
                ttl=86400 * 30
            )
        
        logger.info("A/B test experiment started", experiment_id=experiment_id)
        return True
    
    def assign_variant(self, experiment_id: str, user_id: str) -> Optional[str]:
        """Assign a user to a variant in an experiment."""
        if experiment_id not in self.experiments:
            return None
        
        experiment = self.experiments[experiment_id]
        if experiment["status"] != ExperimentStatus.RUNNING:
            return None
        
        # Check if user already assigned
        if user_id in self.user_assignments[experiment_id]:
            return self.user_assignments[experiment_id][user_id]
        
        # Get variants and weights
        config = experiment["config"]
        variants = config["variants"]
        
        # Determine allocation strategy
        allocation_strategy = TrafficAllocation(config.get("traffic_allocation", "equal"))
        
        if allocation_strategy == TrafficAllocation.EQUAL:
            # Equal allocation
            variant_id = random.choice([v["id"] for v in variants])
            
        elif allocation_strategy == TrafficAllocation.WEIGHTED:
            # Weighted allocation
            weights = [v["traffic_weight"] for v in variants]
            variant_ids = [v["id"] for v in variants]
            variant_id = np.random.choice(variant_ids, p=weights)
            
        else:
            # For other strategies, fall back to weighted for now
            weights = [v["traffic_weight"] for v in variants]
            variant_ids = [v["id"] for v in variants]
            variant_id = np.random.choice(variant_ids, p=weights)
        
        # Store assignment
        self.user_assignments[experiment_id][user_id] = variant_id
        
        return variant_id
    
    async def record_metric(
        self,
        experiment_id: str,
        user_id: str,
        metric_name: str,
        value: Union[float, int, bool]
    ) -> bool:
        """Record a metric value for a user in an experiment."""
        if experiment_id not in self.experiments:
            return False
        
        # Get user's variant assignment
        variant_id = self.assign_variant(experiment_id, user_id)
        if not variant_id:
            return False
        
        # Store metric data
        if experiment_id not in self.experiment_data:
            self.experiment_data[experiment_id] = {}
        
        if variant_id not in self.experiment_data[experiment_id]:
            self.experiment_data[experiment_id][variant_id] = []
        
        # Record the metric
        metric_record = {
            "user_id": user_id,
            "metric_name": metric_name,
            "value": value,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        self.experiment_data[experiment_id][variant_id].append(metric_record)
        
        # Update participant count
        unique_users = set()
        for variant_data in self.experiment_data[experiment_id].values():
            unique_users.update(record["user_id"] for record in variant_data)
        
        self.experiments[experiment_id]["participant_count"] = len(unique_users)
        
        # Cache update
        if self.cache:
            await self.cache.lpush(
                f"experiment_data:{experiment_id}:{variant_id}",
                json.dumps(metric_record),
                max_length=10000
            )
        
        return True
    
    async def analyze_experiment(self, experiment_id: str) -> Optional[ExperimentResult]:
        """Analyze experiment results with statistical tests."""
        if experiment_id not in self.experiments:
            return None
        
        experiment = self.experiments[experiment_id]
        config = experiment["config"]
        metrics_config = config["metrics"]
        
        # Get experiment data
        experiment_data = self.experiment_data.get(experiment_id, {})
        if len(experiment_data) < 2:
            return None
        
        variant_results = {}
        statistical_tests = {}
        confidence_intervals = {}
        significance = {}
        effect_sizes = {}
        recommendations = []
        
        # Process each metric
        for metric_config in metrics_config:
            metric_name = metric_config["name"]
            metric_type = ExperimentType(metric_config["metric_type"])
            
            # Extract metric data for each variant
            variant_data = {}
            for variant_id, records in experiment_data.items():
                metric_values = [
                    record["value"] for record in records 
                    if record["metric_name"] == metric_name
                ]
                
                if metric_values:
                    variant_data[variant_id] = np.array(metric_values)
            
            if len(variant_data) < 2:
                continue
            
            # Find control variant
            control_variant_id = None
            for variant in config["variants"]:
                if variant.get("is_control", False):
                    control_variant_id = variant["id"]
                    break
            
            if control_variant_id is None:
                control_variant_id = list(variant_data.keys())[0]
            
            control_data = variant_data.get(control_variant_id)
            if control_data is None:
                continue
            
            # Analyze against each treatment variant
            for variant_id, treatment_data in variant_data.items():
                if variant_id == control_variant_id:
                    continue
                
                # Check assumptions
                assumptions = self.statistical_analyzer.check_assumptions(
                    control_data, treatment_data, metric_type
                )
                
                # Select appropriate test
                test_type = self.statistical_analyzer.select_statistical_test(
                    metric_type, assumptions
                )
                
                # Perform statistical test
                test_result = self.statistical_analyzer.perform_statistical_test(
                    control_data, treatment_data, test_type
                )
                
                # Calculate confidence intervals
                control_ci = self.statistical_analyzer.calculate_confidence_interval(
                    control_data, 0.95, metric_type
                )
                treatment_ci = self.statistical_analyzer.calculate_confidence_interval(
                    treatment_data, 0.95, metric_type
                )
                
                # Calculate effect size
                effect_size = self.statistical_analyzer.calculate_effect_size(
                    control_data, treatment_data, metric_type
                )
                
                # Store results
                comparison_key = f"{metric_name}_{control_variant_id}_vs_{variant_id}"
                
                statistical_tests[comparison_key] = test_result
                confidence_intervals[metric_name] = {
                    control_variant_id: control_ci,
                    variant_id: treatment_ci
                }
                significance[comparison_key] = test_result["significant"]
                effect_sizes[comparison_key] = effect_size
                
                # Store variant summary statistics
                if metric_name not in variant_results:
                    variant_results[metric_name] = {}
                
                variant_results[metric_name][control_variant_id] = {
                    "mean": float(np.mean(control_data)),
                    "std": float(np.std(control_data)),
                    "count": len(control_data),
                    "median": float(np.median(control_data))
                }
                
                variant_results[metric_name][variant_id] = {
                    "mean": float(np.mean(treatment_data)),
                    "std": float(np.std(treatment_data)),
                    "count": len(treatment_data),
                    "median": float(np.median(treatment_data))
                }
        
        # Generate recommendations
        if statistical_tests:
            significant_results = [k for k, v in significance.items() if v]
            
            if significant_results:
                recommendations.append(f"Found {len(significant_results)} statistically significant results")
                
                # Find best performing variants
                for result_key in significant_results:
                    effect_size = effect_sizes.get(result_key, 0)
                    if effect_size > 0.1:  # Meaningful effect size
                        recommendations.append(f"Strong positive effect detected in {result_key}")
                    elif effect_size < -0.1:
                        recommendations.append(f"Strong negative effect detected in {result_key}")
            else:
                recommendations.append("No statistically significant differences found")
                
                # Check if we need more data
                total_participants = experiment["participant_count"]
                required_sample = config.get("min_sample_size", 1000)
                
                if total_participants < required_sample:
                    recommendations.append(f"Consider collecting more data (current: {total_participants}, required: {required_sample})")
        
        result = ExperimentResult(
            experiment_id=experiment_id,
            variant_results=variant_results,
            statistical_tests=statistical_tests,
            confidence_intervals=confidence_intervals,
            significance=significance,
            effect_sizes=effect_sizes,
            recommendations=recommendations
        )
        
        # Store results
        self.experiments[experiment_id]["results"] = result.to_dict()
        
        # Cache results
        if self.cache:
            await self.cache.set(
                f"experiment_results:{experiment_id}",
                json.dumps(result.to_dict()),
                ttl=86400 * 7  # 7 days
            )
        
        logger.info("A/B test experiment analyzed", experiment_id=experiment_id, 
                   significant_results=len([k for k, v in significance.items() if v]))
        
        return result
    
    async def stop_experiment(self, experiment_id: str, reason: str = "") -> bool:
        """Stop an A/B test experiment."""
        if experiment_id not in self.experiments:
            return False
        
        experiment = self.experiments[experiment_id]
        if experiment["status"] not in [ExperimentStatus.RUNNING, ExperimentStatus.PAUSED]:
            return False
        
        # Analyze final results
        final_results = await self.analyze_experiment(experiment_id)
        
        experiment["status"] = ExperimentStatus.COMPLETED
        experiment["ended_at"] = datetime.utcnow().isoformat()
        experiment["stop_reason"] = reason
        
        if experiment_id in self.active_experiments:
            self.active_experiments.remove(experiment_id)
        
        # Update cache
        if self.cache:
            await self.cache.set(
                f"experiment:{experiment_id}",
                json.dumps(experiment, default=str),
                ttl=86400 * 30
            )
        
        logger.info("A/B test experiment stopped", experiment_id=experiment_id, reason=reason)
        return True
    
    def get_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Get experiment details."""
        return self.experiments.get(experiment_id)
    
    def list_experiments(self, status: Optional[ExperimentStatus] = None) -> List[Dict[str, Any]]:
        """List experiments, optionally filtered by status."""
        experiments = list(self.experiments.values())
        
        if status:
            experiments = [e for e in experiments if e["status"] == status.value]
        
        return experiments
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get system-wide A/B testing metrics."""
        total_experiments = len(self.experiments)
        active_experiments = len(self.active_experiments)
        
        status_counts = defaultdict(int)
        total_participants = 0
        
        for experiment in self.experiments.values():
            status_counts[experiment["status"]] += 1
            total_participants += experiment.get("participant_count", 0)
        
        return {
            "total_experiments": total_experiments,
            "active_experiments": active_experiments,
            "total_participants": total_participants,
            "status_distribution": dict(status_counts),
            "avg_participants_per_experiment": total_participants / total_experiments if total_experiments > 0 else 0
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform A/B testing system health check."""
        try:
            # Create a test experiment
            test_config = ExperimentConfig(
                name="Health Check Test",
                description="Test experiment for health check",
                variants=[
                    ExperimentVariant(id="control", name="Control", is_control=True, traffic_weight=0.5),
                    ExperimentVariant(id="treatment", name="Treatment", traffic_weight=0.5)
                ],
                metrics=[
                    ExperimentMetric(name="test_metric", metric_type=ExperimentType.CONVERSION, primary=True)
                ]
            )
            
            test_experiment_id = await self.create_experiment(test_config)
            await self.start_experiment(test_experiment_id)
            
            # Record some test data
            for i in range(50):
                user_id = f"test_user_{i}"
                await self.record_metric(test_experiment_id, user_id, "test_metric", random.choice([0, 1]))
            
            # Analyze results
            results = await self.analyze_experiment(test_experiment_id)
            
            # Clean up
            await self.stop_experiment(test_experiment_id, "Health check completed")
            del self.experiments[test_experiment_id]
            del self.experiment_data[test_experiment_id]
            del self.user_assignments[test_experiment_id]
            
            return {
                "status": "healthy",
                "test_experiment_created": test_experiment_id is not None,
                "test_analysis_completed": results is not None,
                "system_metrics": self.get_system_metrics()
            }
            
        except Exception as e:
            logger.error("A/B testing health check failed", error=str(e))
            return {
                "status": "unhealthy",
                "error": str(e),
                "system_metrics": self.get_system_metrics()
            }