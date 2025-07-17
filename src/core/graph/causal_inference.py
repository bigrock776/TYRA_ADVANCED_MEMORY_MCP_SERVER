"""
Causal Relationship Inference Engine for Advanced Knowledge Discovery.

This module provides comprehensive causal inference capabilities using local algorithms,
including Granger causality testing, temporal correlation analysis, Pearl's causal hierarchy,
and interventional analysis. All processing is performed locally with zero external API calls.
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Set, Any, Tuple, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque, Counter
import json
import math
import itertools
import warnings

# Statistical and causal inference imports - all local
import networkx as nx
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, accuracy_score
from scipy import stats
from scipy.optimize import minimize
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.stats.contingency_tables import mcnemar

import structlog
from pydantic import BaseModel, Field, ConfigDict

from ...models.memory import Memory
from ...core.cache.redis_cache import RedisCache
from ...core.utils.config import settings
from .reasoning_engine import ReasoningNode, ReasoningEdge, ReasoningPath

logger = structlog.get_logger(__name__)


class CausalRelationType(str, Enum):
    """Types of causal relationships."""
    DIRECT_CAUSE = "direct_cause"                    # A directly causes B
    INDIRECT_CAUSE = "indirect_cause"                # A causes B through intermediates
    COMMON_CAUSE = "common_cause"                    # A and B share common cause
    COMMON_EFFECT = "common_effect"                  # A and B cause common effect
    BIDIRECTIONAL = "bidirectional"                  # A and B cause each other
    SPURIOUS = "spurious"                            # Correlation without causation
    CONFOUNDED = "confounded"                        # Relationship confounded by third variable
    TEMPORAL_PRECEDENCE = "temporal_precedence"      # A precedes B in time


class CausalInferenceMethod(str, Enum):
    """Methods for causal inference."""
    GRANGER_CAUSALITY = "granger_causality"          # Granger causality testing
    PEARL_CAUSAL_HIERARCHY = "pearl_causal_hierarchy"  # Pearl's causal framework
    PROPENSITY_SCORE = "propensity_score"            # Propensity score matching
    INSTRUMENTAL_VARIABLES = "instrumental_variables"  # IV estimation
    REGRESSION_DISCONTINUITY = "regression_discontinuity"  # RD design
    DIFFERENCE_IN_DIFFERENCES = "difference_in_differences"  # DiD
    NATURAL_EXPERIMENT = "natural_experiment"         # Natural experiments
    CORRELATION_ANALYSIS = "correlation_analysis"     # Basic correlation


class ConfidenceLevel(str, Enum):
    """Confidence levels for causal claims."""
    VERY_HIGH = "very_high"      # 95%+ confidence
    HIGH = "high"                # 80-95% confidence
    MODERATE = "moderate"        # 60-80% confidence
    LOW = "low"                  # 40-60% confidence
    VERY_LOW = "very_low"        # <40% confidence


@dataclass
class CausalEvidence:
    """Evidence supporting a causal claim."""
    evidence_type: str
    strength: float  # 0.0 to 1.0
    p_value: Optional[float] = None
    effect_size: Optional[float] = None
    test_statistic: Optional[float] = None
    method_used: str = ""
    temporal_order: bool = False
    confounders_controlled: List[str] = field(default_factory=list)
    sample_size: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CausalClaim:
    """Represents a causal claim between entities."""
    cause_entity: str
    effect_entity: str
    relationship_type: CausalRelationType
    confidence: float
    confidence_level: ConfidenceLevel
    evidence: List[CausalEvidence] = field(default_factory=list)
    mediating_variables: List[str] = field(default_factory=list)
    confounding_variables: List[str] = field(default_factory=list)
    effect_size: Optional[float] = None
    time_lag: Optional[timedelta] = None
    discovered_at: datetime = field(default_factory=datetime.utcnow)
    
    def get_overall_evidence_strength(self) -> float:
        """Calculate overall strength of evidence."""
        if not self.evidence:
            return 0.0
        
        # Weight different types of evidence
        weights = {
            'experimental': 1.0,
            'quasi_experimental': 0.8,
            'granger': 0.6,
            'correlation': 0.3,
            'temporal': 0.4
        }
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for evidence in self.evidence:
            weight = weights.get(evidence.evidence_type, 0.5)
            weighted_sum += evidence.strength * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def is_temporally_consistent(self) -> bool:
        """Check if causal claim is temporally consistent."""
        return any(evidence.temporal_order for evidence in self.evidence)


@dataclass
class CausalGraph:
    """Represents a causal graph structure."""
    nodes: Dict[str, ReasoningNode]
    edges: Dict[Tuple[str, str], CausalClaim]
    graph: nx.DiGraph = field(default_factory=nx.DiGraph)
    
    def add_causal_claim(self, claim: CausalClaim):
        """Add a causal claim to the graph."""
        self.edges[(claim.cause_entity, claim.effect_entity)] = claim
        self.graph.add_edge(
            claim.cause_entity,
            claim.effect_entity,
            weight=claim.confidence,
            relationship_type=claim.relationship_type.value
        )
    
    def get_causal_paths(self, source: str, target: str, max_length: int = 5) -> List[List[str]]:
        """Find causal paths between entities."""
        try:
            paths = list(nx.all_simple_paths(
                self.graph, source, target, cutoff=max_length
            ))
            return paths
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []
    
    def detect_cycles(self) -> List[List[str]]:
        """Detect causal cycles in the graph."""
        try:
            cycles = list(nx.simple_cycles(self.graph))
            return cycles
        except nx.NetworkXNoCycle:
            return []
    
    def find_confounders(self, cause: str, effect: str) -> List[str]:
        """Find potential confounding variables."""
        confounders = []
        
        # A confounder influences both cause and effect
        for node in self.graph.nodes():
            if node != cause and node != effect:
                has_path_to_cause = nx.has_path(self.graph, node, cause)
                has_path_to_effect = nx.has_path(self.graph, node, effect)
                
                if has_path_to_cause and has_path_to_effect:
                    confounders.append(node)
        
        return confounders


class GrangerCausalityAnalyzer:
    """
    Granger Causality Analysis Engine.
    
    Tests whether past values of X help predict Y better than past values of Y alone.
    """
    
    def __init__(self, max_lags: int = 10, significance_level: float = 0.05):
        self.max_lags = max_lags
        self.significance_level = significance_level
    
    async def analyze_causality(
        self,
        cause_series: pd.Series,
        effect_series: pd.Series,
        entity_names: Tuple[str, str]
    ) -> Optional[CausalClaim]:
        """Perform Granger causality analysis."""
        
        try:
            # Ensure series are aligned and have datetime index
            if len(cause_series) != len(effect_series):
                logger.warning(f"Series length mismatch: {len(cause_series)} vs {len(effect_series)}")
                return None
            
            if len(cause_series) < self.max_lags + 10:
                logger.warning(f"Insufficient data for Granger test: {len(cause_series)} points")
                return None
            
            # Prepare data for Granger test
            data = pd.DataFrame({
                'effect': effect_series.values,
                'cause': cause_series.values
            })
            
            # Handle missing values
            data = data.dropna()
            if len(data) < self.max_lags + 10:
                logger.warning("Insufficient data after removing NaN values")
                return None
            
            # Perform Granger causality test
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                test_results = grangercausalitytests(
                    data[['effect', 'cause']], 
                    maxlag=min(self.max_lags, len(data) // 4),
                    verbose=False
                )
            
            # Extract best lag and test statistics
            best_lag, best_p_value, best_f_stat = self._extract_best_result(test_results)
            
            if best_p_value is None:
                return None
            
            # Calculate confidence and effect size
            confidence = 1.0 - best_p_value
            effect_size = self._calculate_effect_size(data, best_lag)
            
            # Create evidence
            evidence = CausalEvidence(
                evidence_type="granger",
                strength=confidence,
                p_value=best_p_value,
                effect_size=effect_size,
                test_statistic=best_f_stat,
                method_used=f"granger_causality_lag_{best_lag}",
                temporal_order=True,
                sample_size=len(data)
            )
            
            # Determine confidence level
            if confidence >= 0.95:
                conf_level = ConfidenceLevel.VERY_HIGH
            elif confidence >= 0.8:
                conf_level = ConfidenceLevel.HIGH
            elif confidence >= 0.6:
                conf_level = ConfidenceLevel.MODERATE
            elif confidence >= 0.4:
                conf_level = ConfidenceLevel.LOW
            else:
                conf_level = ConfidenceLevel.VERY_LOW
            
            # Create causal claim
            claim = CausalClaim(
                cause_entity=entity_names[0],
                effect_entity=entity_names[1],
                relationship_type=CausalRelationType.DIRECT_CAUSE,
                confidence=confidence,
                confidence_level=conf_level,
                evidence=[evidence],
                effect_size=effect_size,
                time_lag=pd.Timedelta(days=best_lag) if best_lag else None
            )
            
            return claim
            
        except Exception as e:
            logger.error(f"Granger causality analysis failed: {e}")
            return None
    
    def _extract_best_result(self, test_results: Dict) -> Tuple[Optional[int], Optional[float], Optional[float]]:
        """Extract the best lag and its statistics."""
        
        best_lag = None
        best_p_value = None
        best_f_stat = None
        
        for lag, result in test_results.items():
            # Get F-test results (first test in the tuple)
            f_test = result[0]
            p_value = f_test['ssr_ftest'][1]  # p-value
            f_stat = f_test['ssr_ftest'][0]   # F-statistic
            
            if best_p_value is None or p_value < best_p_value:
                best_lag = lag
                best_p_value = p_value
                best_f_stat = f_stat
        
        return best_lag, best_p_value, best_f_stat
    
    def _calculate_effect_size(self, data: pd.DataFrame, lag: int) -> float:
        """Calculate effect size for Granger causality."""
        
        try:
            # Create lagged variables
            lagged_data = data.copy()
            for i in range(1, lag + 1):
                lagged_data[f'cause_lag_{i}'] = data['cause'].shift(i)
                lagged_data[f'effect_lag_{i}'] = data['effect'].shift(i)
            
            # Remove rows with NaN values
            lagged_data = lagged_data.dropna()
            
            if len(lagged_data) < 10:
                return 0.0
            
            # Fit restricted model (only lagged effects)
            y = lagged_data['effect']
            X_restricted = lagged_data[[f'effect_lag_{i}' for i in range(1, lag + 1)]]
            X_restricted = sm.add_constant(X_restricted)
            
            model_restricted = sm.OLS(y, X_restricted).fit()
            ssr_restricted = model_restricted.ssr
            
            # Fit unrestricted model (lagged effects + lagged causes)
            X_unrestricted = lagged_data[[f'effect_lag_{i}' for i in range(1, lag + 1)] + 
                                       [f'cause_lag_{i}' for i in range(1, lag + 1)]]
            X_unrestricted = sm.add_constant(X_unrestricted)
            
            model_unrestricted = sm.OLS(y, X_unrestricted).fit()
            ssr_unrestricted = model_unrestricted.ssr
            
            # Calculate effect size as proportional reduction in error
            effect_size = (ssr_restricted - ssr_unrestricted) / ssr_restricted
            
            return max(0.0, min(1.0, effect_size))
            
        except Exception as e:
            logger.warning(f"Effect size calculation failed: {e}")
            return 0.0


class PearlCausalAnalyzer:
    """
    Pearl's Causal Hierarchy Implementation.
    
    Implements the three levels of Pearl's causal hierarchy:
    1. Association (seeing): P(Y|X)
    2. Intervention (doing): P(Y|do(X))
    3. Counterfactuals (imagining): P(Y_x|X',Y')
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
    
    async def analyze_causal_hierarchy(
        self,
        data: pd.DataFrame,
        cause_var: str,
        effect_var: str,
        confounders: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Analyze causality using Pearl's hierarchy."""
        
        results = {
            'association': await self._analyze_association(data, cause_var, effect_var),
            'intervention': await self._estimate_intervention(data, cause_var, effect_var, confounders),
            'counterfactual': await self._estimate_counterfactual(data, cause_var, effect_var, confounders)
        }
        
        return results
    
    async def _analyze_association(
        self,
        data: pd.DataFrame,
        cause_var: str,
        effect_var: str
    ) -> Dict[str, float]:
        """Level 1: Association analysis."""
        
        try:
            # Calculate correlation
            correlation = data[cause_var].corr(data[effect_var])
            
            # Calculate mutual information (for non-linear relationships)
            mutual_info = self._calculate_mutual_information(
                data[cause_var].values,
                data[effect_var].values
            )
            
            # Fit linear regression
            X = data[[cause_var]].values
            y = data[effect_var].values
            
            # Remove NaN values
            mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
            X_clean = X[mask]
            y_clean = y[mask]
            
            if len(X_clean) < 10:
                return {'correlation': 0.0, 'r_squared': 0.0, 'mutual_information': 0.0}
            
            X_scaled = self.scaler.fit_transform(X_clean)
            
            reg = LinearRegression().fit(X_scaled, y_clean)
            r_squared = reg.score(X_scaled, y_clean)
            
            return {
                'correlation': correlation,
                'r_squared': r_squared,
                'mutual_information': mutual_info,
                'coefficient': reg.coef_[0] if len(reg.coef_) > 0 else 0.0
            }
            
        except Exception as e:
            logger.error(f"Association analysis failed: {e}")
            return {'correlation': 0.0, 'r_squared': 0.0, 'mutual_information': 0.0}
    
    async def _estimate_intervention(
        self,
        data: pd.DataFrame,
        cause_var: str,
        effect_var: str,
        confounders: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """Level 2: Intervention estimation using backdoor adjustment."""
        
        try:
            if confounders is None:
                confounders = []
            
            # Backdoor adjustment: P(Y|do(X)) = Σ_Z P(Y|X,Z) * P(Z)
            available_confounders = [c for c in confounders if c in data.columns]
            
            if not available_confounders:
                # No confounders available, use simple regression
                return await self._simple_intervention_estimate(data, cause_var, effect_var)
            
            # Create treatment and control groups
            cause_median = data[cause_var].median()
            treatment_mask = data[cause_var] > cause_median
            
            # Estimate ATE (Average Treatment Effect)
            ate = await self._calculate_ate_with_confounders(
                data, treatment_mask, effect_var, available_confounders
            )
            
            # Estimate effect using regression adjustment
            regression_effect = await self._regression_adjustment(
                data, cause_var, effect_var, available_confounders
            )
            
            return {
                'average_treatment_effect': ate,
                'regression_adjusted_effect': regression_effect,
                'confounders_used': len(available_confounders)
            }
            
        except Exception as e:
            logger.error(f"Intervention estimation failed: {e}")
            return {'average_treatment_effect': 0.0, 'regression_adjusted_effect': 0.0}
    
    async def _estimate_counterfactual(
        self,
        data: pd.DataFrame,
        cause_var: str,
        effect_var: str,
        confounders: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """Level 3: Counterfactual estimation."""
        
        try:
            # Simplified counterfactual estimation
            # In practice, this would require structural causal models
            
            # Calculate what would have happened if treatment was different
            cause_values = data[cause_var].values
            effect_values = data[effect_var].values
            
            # Remove NaN values
            mask = ~(np.isnan(cause_values) | np.isnan(effect_values))
            cause_clean = cause_values[mask]
            effect_clean = effect_values[mask]
            
            if len(cause_clean) < 10:
                return {'counterfactual_effect': 0.0, 'individual_treatment_effects': []}
            
            # Fit a model to predict effects
            X = cause_clean.reshape(-1, 1)
            if confounders:
                available_confounders = [c for c in confounders if c in data.columns]
                if available_confounders:
                    confounder_data = data[available_confounders].values[mask]
                    X = np.column_stack([X, confounder_data])
            
            reg = RandomForestRegressor(n_estimators=50, random_state=42)
            reg.fit(X, effect_clean)
            
            # Calculate individual treatment effects
            # ITE = Y(1) - Y(0) for each individual
            individual_effects = []
            
            for i in range(min(100, len(X))):  # Limit to 100 for performance
                x_original = X[i:i+1]
                
                # Predict under current treatment
                y_factual = reg.predict(x_original)[0]
                
                # Predict under alternative treatment
                x_counterfactual = x_original.copy()
                x_counterfactual[0, 0] = 1 - x_counterfactual[0, 0]  # Flip treatment
                y_counterfactual = reg.predict(x_counterfactual)[0]
                
                ite = y_factual - y_counterfactual
                individual_effects.append(ite)
            
            avg_counterfactual_effect = np.mean(individual_effects) if individual_effects else 0.0
            
            return {
                'counterfactual_effect': avg_counterfactual_effect,
                'individual_treatment_effects': individual_effects[:10]  # Return first 10
            }
            
        except Exception as e:
            logger.error(f"Counterfactual estimation failed: {e}")
            return {'counterfactual_effect': 0.0, 'individual_treatment_effects': []}
    
    def _calculate_mutual_information(self, x: np.ndarray, y: np.ndarray) -> float:
        """Calculate mutual information between two variables."""
        
        try:
            # Discretize continuous variables
            x_bins = np.histogram_bin_edges(x, bins='auto')
            y_bins = np.histogram_bin_edges(y, bins='auto')
            
            x_discrete = np.digitize(x, x_bins)
            y_discrete = np.digitize(y, y_bins)
            
            # Calculate joint and marginal distributions
            joint_hist, _, _ = np.histogram2d(x_discrete, y_discrete)
            joint_hist = joint_hist / joint_hist.sum()
            
            x_hist = joint_hist.sum(axis=1)
            y_hist = joint_hist.sum(axis=0)
            
            # Calculate mutual information
            mi = 0.0
            for i in range(len(x_hist)):
                for j in range(len(y_hist)):
                    if joint_hist[i, j] > 0 and x_hist[i] > 0 and y_hist[j] > 0:
                        mi += joint_hist[i, j] * np.log(
                            joint_hist[i, j] / (x_hist[i] * y_hist[j])
                        )
            
            return mi
            
        except Exception as e:
            logger.warning(f"Mutual information calculation failed: {e}")
            return 0.0
    
    async def _simple_intervention_estimate(
        self,
        data: pd.DataFrame,
        cause_var: str,
        effect_var: str
    ) -> Dict[str, float]:
        """Simple intervention estimate without confounders."""
        
        # Calculate treatment effect as difference in means
        cause_median = data[cause_var].median()
        
        treatment_group = data[data[cause_var] > cause_median][effect_var]
        control_group = data[data[cause_var] <= cause_median][effect_var]
        
        treatment_mean = treatment_group.mean() if len(treatment_group) > 0 else 0.0
        control_mean = control_group.mean() if len(control_group) > 0 else 0.0
        
        ate = treatment_mean - control_mean
        
        return {
            'average_treatment_effect': ate,
            'regression_adjusted_effect': ate,
            'confounders_used': 0
        }
    
    async def _calculate_ate_with_confounders(
        self,
        data: pd.DataFrame,
        treatment_mask: pd.Series,
        effect_var: str,
        confounders: List[str]
    ) -> float:
        """Calculate ATE with confounder adjustment."""
        
        try:
            # Use propensity score matching approximation
            # Fit propensity score model
            X = data[confounders].values
            y = treatment_mask.values
            
            # Remove NaN values
            mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
            X_clean = X[mask]
            y_clean = y[mask]
            
            if len(X_clean) < 10:
                return 0.0
            
            ps_model = LogisticRegression()
            ps_model.fit(X_clean, y_clean)
            
            # Calculate propensity scores
            propensity_scores = ps_model.predict_proba(X_clean)[:, 1]
            
            # Weight observations by inverse propensity scores
            treatment_indices = np.where(y_clean)[0]
            control_indices = np.where(~y_clean)[0]
            
            if len(treatment_indices) == 0 or len(control_indices) == 0:
                return 0.0
            
            # Calculate weighted outcomes
            effect_values = data[effect_var].values[mask]
            
            treatment_outcomes = effect_values[treatment_indices]
            treatment_weights = 1.0 / propensity_scores[treatment_indices]
            treatment_weights = np.clip(treatment_weights, 0.1, 10.0)  # Clip extreme weights
            
            control_outcomes = effect_values[control_indices]
            control_weights = 1.0 / (1.0 - propensity_scores[control_indices])
            control_weights = np.clip(control_weights, 0.1, 10.0)
            
            # Calculate weighted means
            weighted_treatment_mean = np.average(treatment_outcomes, weights=treatment_weights)
            weighted_control_mean = np.average(control_outcomes, weights=control_weights)
            
            ate = weighted_treatment_mean - weighted_control_mean
            
            return ate
            
        except Exception as e:
            logger.warning(f"ATE calculation with confounders failed: {e}")
            return 0.0
    
    async def _regression_adjustment(
        self,
        data: pd.DataFrame,
        cause_var: str,
        effect_var: str,
        confounders: List[str]
    ) -> float:
        """Estimate causal effect using regression adjustment."""
        
        try:
            # Prepare data
            feature_vars = [cause_var] + confounders
            X = data[feature_vars].values
            y = data[effect_var].values
            
            # Remove NaN values
            mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
            X_clean = X[mask]
            y_clean = y[mask]
            
            if len(X_clean) < 10:
                return 0.0
            
            # Fit regression model
            reg = LinearRegression()
            reg.fit(X_clean, y_clean)
            
            # The coefficient of the cause variable is the causal effect
            causal_effect = reg.coef_[0]  # First coefficient is for cause_var
            
            return causal_effect
            
        except Exception as e:
            logger.warning(f"Regression adjustment failed: {e}")
            return 0.0


class CausalInferenceEngine:
    """
    Main Causal Inference Engine.
    
    Coordinates different causal inference methods and maintains causal graph.
    """
    
    def __init__(
        self,
        redis_cache: Optional[RedisCache] = None,
        cache_ttl: int = 3600
    ):
        self.redis_cache = redis_cache
        self.cache_ttl = cache_ttl
        
        # Initialize analyzers
        self.granger_analyzer = GrangerCausalityAnalyzer()
        self.pearl_analyzer = PearlCausalAnalyzer()
        
        # Causal graph
        self.causal_graph = CausalGraph(nodes={}, edges={})
        
        # Statistics
        self.stats = {
            'total_analyses': 0,
            'successful_inferences': 0,
            'granger_tests': 0,
            'pearl_analyses': 0,
            'claims_generated': 0
        }
    
    async def infer_causality(
        self,
        memories: List[Memory],
        entity_pairs: Optional[List[Tuple[str, str]]] = None,
        methods: Optional[List[CausalInferenceMethod]] = None
    ) -> List[CausalClaim]:
        """
        Infer causal relationships from memory data.
        
        Args:
            memories: List of memories to analyze
            entity_pairs: Specific entity pairs to test (optional)
            methods: Inference methods to use (optional)
        
        Returns:
            List of causal claims with evidence
        """
        
        start_time = datetime.utcnow()
        
        if methods is None:
            methods = [
                CausalInferenceMethod.GRANGER_CAUSALITY,
                CausalInferenceMethod.CORRELATION_ANALYSIS,
                CausalInferenceMethod.PEARL_CAUSAL_HIERARCHY
            ]
        
        # Extract time series data from memories
        time_series_data = await self._extract_time_series(memories)
        
        if not time_series_data:
            logger.warning("No time series data found in memories")
            return []
        
        # Auto-detect entity pairs if not provided
        if entity_pairs is None:
            entity_pairs = await self._auto_detect_entity_pairs(time_series_data)
        
        claims = []
        
        for cause_entity, effect_entity in entity_pairs:
            if cause_entity not in time_series_data or effect_entity not in time_series_data:
                continue
            
            self.stats['total_analyses'] += 1
            
            # Cache key for this analysis
            cache_key = f"causal_inference:{cause_entity}:{effect_entity}:{hash(str(sorted(methods)))}"
            
            # Check cache
            if self.redis_cache:
                cached_result = await self.redis_cache.get(cache_key)
                if cached_result:
                    claims.append(CausalClaim(**cached_result))
                    continue
            
            claim = await self._analyze_entity_pair(
                cause_entity,
                effect_entity,
                time_series_data,
                methods
            )
            
            if claim:
                claims.append(claim)
                self.stats['successful_inferences'] += 1
                self.stats['claims_generated'] += 1
                
                # Add to causal graph
                self.causal_graph.add_causal_claim(claim)
                
                # Cache result
                if self.redis_cache:
                    await self.redis_cache.set(
                        cache_key,
                        claim.__dict__,
                        ttl=self.cache_ttl
                    )
        
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        logger.info(
            f"Causal inference completed",
            processing_time=processing_time,
            claims_found=len(claims),
            entity_pairs_analyzed=len(entity_pairs)
        )
        
        return claims
    
    async def _extract_time_series(self, memories: List[Memory]) -> Dict[str, pd.Series]:
        """Extract time series data from memories."""
        
        time_series = defaultdict(list)
        
        for memory in memories:
            # Extract timestamp
            timestamp = memory.created_at
            
            # Extract entities and their values
            # This is a simplified extraction - in practice, you'd use NER
            content = memory.content.lower()
            
            # Look for numeric patterns
            import re
            numeric_patterns = re.findall(r'(\w+)\s*[:=]\s*([+-]?\d*\.?\d+)', content)
            
            for entity, value in numeric_patterns:
                try:
                    numeric_value = float(value)
                    time_series[entity].append((timestamp, numeric_value))
                except ValueError:
                    continue
            
            # Look for boolean/categorical patterns
            boolean_patterns = re.findall(r'(\w+)\s*(?:is|was|became)\s*(true|false|yes|no|on|off)', content)
            
            for entity, value in boolean_patterns:
                numeric_value = 1.0 if value.lower() in ['true', 'yes', 'on'] else 0.0
                time_series[entity].append((timestamp, numeric_value))
        
        # Convert to pandas Series
        series_dict = {}
        for entity, data_points in time_series.items():
            if len(data_points) < 5:  # Need minimum data points
                continue
            
            # Sort by timestamp
            data_points.sort(key=lambda x: x[0])
            
            timestamps = [dp[0] for dp in data_points]
            values = [dp[1] for dp in data_points]
            
            series = pd.Series(values, index=pd.DatetimeIndex(timestamps))
            series_dict[entity] = series
        
        return series_dict
    
    async def _auto_detect_entity_pairs(
        self,
        time_series_data: Dict[str, pd.Series]
    ) -> List[Tuple[str, str]]:
        """Auto-detect interesting entity pairs for causal analysis."""
        
        entities = list(time_series_data.keys())
        pairs = []
        
        # Generate all pairs
        for i, entity1 in enumerate(entities):
            for entity2 in entities[i+1:]:
                # Basic correlation filter
                try:
                    correlation = time_series_data[entity1].corr(time_series_data[entity2])
                    if abs(correlation) > 0.1:  # Minimum correlation threshold
                        pairs.append((entity1, entity2))
                        pairs.append((entity2, entity1))  # Both directions
                except Exception:
                    continue
        
        # Limit to top pairs by correlation strength
        pair_correlations = []
        for entity1, entity2 in pairs:
            try:
                correlation = abs(time_series_data[entity1].corr(time_series_data[entity2]))
                pair_correlations.append(((entity1, entity2), correlation))
            except Exception:
                continue
        
        # Sort by correlation and take top pairs
        pair_correlations.sort(key=lambda x: x[1], reverse=True)
        top_pairs = [pair for pair, _ in pair_correlations[:20]]  # Top 20 pairs
        
        return top_pairs
    
    async def _analyze_entity_pair(
        self,
        cause_entity: str,
        effect_entity: str,
        time_series_data: Dict[str, pd.Series],
        methods: List[CausalInferenceMethod]
    ) -> Optional[CausalClaim]:
        """Analyze a specific entity pair for causal relationships."""
        
        cause_series = time_series_data[cause_entity]
        effect_series = time_series_data[effect_entity]
        
        # Align series
        common_index = cause_series.index.intersection(effect_series.index)
        if len(common_index) < 5:
            return None
        
        cause_aligned = cause_series.loc[common_index]
        effect_aligned = effect_series.loc[common_index]
        
        evidence_list = []
        
        # Apply each method
        for method in methods:
            if method == CausalInferenceMethod.GRANGER_CAUSALITY:
                claim = await self.granger_analyzer.analyze_causality(
                    cause_aligned, effect_aligned, (cause_entity, effect_entity)
                )
                if claim and claim.evidence:
                    evidence_list.extend(claim.evidence)
                    self.stats['granger_tests'] += 1
            
            elif method == CausalInferenceMethod.PEARL_CAUSAL_HIERARCHY:
                data = pd.DataFrame({
                    cause_entity: cause_aligned.values,
                    effect_entity: effect_aligned.values
                })
                
                pearl_results = await self.pearl_analyzer.analyze_causal_hierarchy(
                    data, cause_entity, effect_entity
                )
                
                # Convert Pearl results to evidence
                if pearl_results.get('association', {}).get('correlation', 0) != 0:
                    evidence = CausalEvidence(
                        evidence_type="correlation",
                        strength=abs(pearl_results['association']['correlation']),
                        method_used="pearl_association",
                        sample_size=len(data)
                    )
                    evidence_list.append(evidence)
                
                if pearl_results.get('intervention', {}).get('average_treatment_effect', 0) != 0:
                    effect_size = abs(pearl_results['intervention']['average_treatment_effect'])
                    evidence = CausalEvidence(
                        evidence_type="quasi_experimental",
                        strength=min(1.0, effect_size),
                        effect_size=effect_size,
                        method_used="pearl_intervention",
                        sample_size=len(data)
                    )
                    evidence_list.append(evidence)
                
                self.stats['pearl_analyses'] += 1
            
            elif method == CausalInferenceMethod.CORRELATION_ANALYSIS:
                correlation = cause_aligned.corr(effect_aligned)
                if abs(correlation) > 0.1:
                    evidence = CausalEvidence(
                        evidence_type="correlation",
                        strength=abs(correlation),
                        method_used="pearson_correlation",
                        sample_size=len(cause_aligned)
                    )
                    evidence_list.append(evidence)
        
        # Create combined causal claim if we have evidence
        if not evidence_list:
            return None
        
        # Calculate overall confidence
        overall_confidence = np.mean([e.strength for e in evidence_list])
        
        # Determine confidence level
        if overall_confidence >= 0.8:
            conf_level = ConfidenceLevel.VERY_HIGH
        elif overall_confidence >= 0.6:
            conf_level = ConfidenceLevel.HIGH
        elif overall_confidence >= 0.4:
            conf_level = ConfidenceLevel.MODERATE
        elif overall_confidence >= 0.2:
            conf_level = ConfidenceLevel.LOW
        else:
            conf_level = ConfidenceLevel.VERY_LOW
        
        # Determine relationship type
        relationship_type = CausalRelationType.DIRECT_CAUSE
        if len(evidence_list) == 1 and evidence_list[0].evidence_type == "correlation":
            relationship_type = CausalRelationType.SPURIOUS
        
        claim = CausalClaim(
            cause_entity=cause_entity,
            effect_entity=effect_entity,
            relationship_type=relationship_type,
            confidence=overall_confidence,
            confidence_level=conf_level,
            evidence=evidence_list
        )
        
        return claim
    
    async def find_confounders(
        self,
        cause: str,
        effect: str,
        candidate_variables: List[str],
        time_series_data: Dict[str, pd.Series]
    ) -> List[str]:
        """Find potential confounding variables."""
        
        confounders = []
        
        for candidate in candidate_variables:
            if candidate == cause or candidate == effect:
                continue
            
            if candidate not in time_series_data:
                continue
            
            candidate_series = time_series_data[candidate]
            cause_series = time_series_data.get(cause)
            effect_series = time_series_data.get(effect)
            
            if cause_series is None or effect_series is None:
                continue
            
            # Check if candidate correlates with both cause and effect
            try:
                cause_corr = abs(candidate_series.corr(cause_series))
                effect_corr = abs(candidate_series.corr(effect_series))
                
                if cause_corr > 0.3 and effect_corr > 0.3:
                    confounders.append(candidate)
            
            except Exception:
                continue
        
        return confounders
    
    async def validate_causal_claims(
        self,
        claims: List[CausalClaim],
        validation_data: Optional[List[Memory]] = None
    ) -> List[CausalClaim]:
        """Validate causal claims using additional data or consistency checks."""
        
        validated_claims = []
        
        for claim in claims:
            # Basic consistency checks
            if not claim.is_temporally_consistent():
                claim.confidence *= 0.8  # Reduce confidence
            
            # Check for contradictory claims
            contradictory = self._find_contradictory_claims(claim, claims)
            if contradictory:
                claim.confidence *= 0.7
            
            # Evidence strength validation
            evidence_strength = claim.get_overall_evidence_strength()
            if evidence_strength < 0.2:
                continue  # Filter out very weak claims
            
            validated_claims.append(claim)
        
        return validated_claims
    
    def _find_contradictory_claims(
        self,
        claim: CausalClaim,
        all_claims: List[CausalClaim]
    ) -> List[CausalClaim]:
        """Find claims that contradict the given claim."""
        
        contradictory = []
        
        for other_claim in all_claims:
            if other_claim == claim:
                continue
            
            # Check for direct contradiction (A->B vs B->A with different relationship types)
            if (claim.cause_entity == other_claim.effect_entity and 
                claim.effect_entity == other_claim.cause_entity and
                claim.relationship_type != other_claim.relationship_type):
                contradictory.append(other_claim)
        
        return contradictory
    
    async def export_causal_graph(self) -> Dict[str, Any]:
        """Export the current causal graph."""
        
        return {
            'nodes': [
                {
                    'id': node_id,
                    'type': node.entity_type,
                    'properties': node.properties,
                    'importance': node.importance_score
                }
                for node_id, node in self.causal_graph.nodes.items()
            ],
            'edges': [
                {
                    'source': claim.cause_entity,
                    'target': claim.effect_entity,
                    'relationship_type': claim.relationship_type.value,
                    'confidence': claim.confidence,
                    'evidence_count': len(claim.evidence),
                    'effect_size': claim.effect_size
                }
                for claim in self.causal_graph.edges.values()
            ],
            'statistics': self.stats.copy()
        }
    
    async def get_causal_explanation(
        self,
        cause: str,
        effect: str,
        max_path_length: int = 3
    ) -> Optional[str]:
        """Generate natural language explanation for causal relationship."""
        
        # Find causal paths
        paths = self.causal_graph.get_causal_paths(cause, effect, max_path_length)
        
        if not paths:
            return None
        
        # Get the shortest path
        shortest_path = min(paths, key=len)
        
        # Build explanation
        if len(shortest_path) == 2:
            # Direct relationship
            claim = self.causal_graph.edges.get((cause, effect))
            if claim:
                confidence_desc = claim.confidence_level.value.replace('_', ' ')
                return f"There is {confidence_desc} confidence that {cause} directly causes {effect} " \
                       f"(confidence: {claim.confidence:.2f})."
        
        else:
            # Indirect relationship
            path_desc = " → ".join(shortest_path)
            return f"{cause} may indirectly influence {effect} through the path: {path_desc}."
        
        return f"A causal relationship exists between {cause} and {effect}."
    
    async def get_engine_stats(self) -> Dict[str, Any]:
        """Get comprehensive engine statistics."""
        
        return {
            'causal_inference_stats': self.stats.copy(),
            'graph_stats': {
                'total_claims': len(self.causal_graph.edges),
                'total_entities': len(self.causal_graph.nodes),
                'cycles_detected': len(self.causal_graph.detect_cycles())
            },
            'analyzer_stats': {
                'granger_analyzer_config': {
                    'max_lags': self.granger_analyzer.max_lags,
                    'significance_level': self.granger_analyzer.significance_level
                }
            }
        }
    
    async def export_causal_model(
        self,
        format: str = "json",
        include_metadata: bool = True
    ) -> Dict[str, Any]:
        """Export the complete causal model for analysis or transfer."""
        try:
            # Build complete causal network
            causal_network = {
                "nodes": [],
                "edges": [],
                "metadata": {}
            }
            
            # Export nodes (entities/variables)
            for node_id in self.causal_graph.graph.nodes():
                node_data = self.causal_graph.graph.nodes[node_id]
                causal_network["nodes"].append({
                    "id": node_id,
                    "type": node_data.get("type", "entity"),
                    "properties": node_data.get("properties", {}),
                    "confidence": node_data.get("confidence", 1.0),
                    "temporal_info": node_data.get("temporal_info", {})
                })
            
            # Export edges (causal relationships)
            for source, target, edge_data in self.causal_graph.graph.edges(data=True):
                causal_network["edges"].append({
                    "source": source,
                    "target": target,
                    "causal_strength": edge_data.get("causal_strength", 0.0),
                    "confidence": edge_data.get("confidence", 0.0),
                    "direction": edge_data.get("direction", "unknown"),
                    "evidence": edge_data.get("evidence", []),
                    "temporal_lag": edge_data.get("temporal_lag", 0),
                    "relationship_type": edge_data.get("relationship_type", "causes")
                })
            
            # Add metadata if requested
            if include_metadata:
                causal_network["metadata"] = {
                    "export_timestamp": datetime.utcnow().isoformat(),
                    "model_version": "1.0",
                    "total_nodes": len(causal_network["nodes"]),
                    "total_edges": len(causal_network["edges"]),
                    "inference_algorithms": ["granger", "pearl_hierarchy"],
                    "confidence_threshold": getattr(self, 'confidence_threshold', 0.5),
                    "temporal_analysis_enabled": True,
                    "statistics": {
                        "avg_causal_strength": np.mean([
                            e["causal_strength"] for e in causal_network["edges"]
                        ]) if causal_network["edges"] else 0,
                        "strong_causal_links": len([
                            e for e in causal_network["edges"] 
                            if e["causal_strength"] > 0.7
                        ]),
                        "temporal_relationships": len([
                            e for e in causal_network["edges"] 
                            if e["temporal_lag"] > 0
                        ])
                    }
                }
            
            # Format output based on requested format
            if format.lower() == "graphml":
                return self._convert_to_graphml(causal_network)
            elif format.lower() == "dot":
                return self._convert_to_dot(causal_network)
            elif format.lower() == "cytoscape":
                return self._convert_to_cytoscape(causal_network)
            else:
                return causal_network
                
        except Exception as e:
            logger.error(f"Causal model export failed: {e}")
            return {"error": str(e)}
    
    async def import_causal_model(
        self,
        model_data: Dict[str, Any],
        merge_strategy: str = "replace"
    ) -> bool:
        """Import a causal model from external data."""
        try:
            if "nodes" not in model_data or "edges" not in model_data:
                raise ValueError("Invalid model format - missing nodes or edges")
            
            # Clear existing graph if replace strategy
            if merge_strategy == "replace":
                self.causal_graph.graph.clear()
            
            # Import nodes
            for node in model_data["nodes"]:
                node_id = node["id"]
                self.causal_graph.graph.add_node(
                    node_id,
                    type=node.get("type", "entity"),
                    properties=node.get("properties", {}),
                    confidence=node.get("confidence", 1.0),
                    temporal_info=node.get("temporal_info", {})
                )
            
            # Import edges
            for edge in model_data["edges"]:
                source = edge["source"]
                target = edge["target"]
                
                # Only add edge if both nodes exist
                if source in self.causal_graph.graph.nodes and target in self.causal_graph.graph.nodes:
                    self.causal_graph.graph.add_edge(
                        source, target,
                        causal_strength=edge.get("causal_strength", 0.0),
                        confidence=edge.get("confidence", 0.0),
                        direction=edge.get("direction", "unknown"),
                        evidence=edge.get("evidence", []),
                        temporal_lag=edge.get("temporal_lag", 0),
                        relationship_type=edge.get("relationship_type", "causes")
                    )
            
            logger.info(
                "Causal model imported successfully",
                nodes_imported=len(model_data["nodes"]),
                edges_imported=len(model_data["edges"]),
                merge_strategy=merge_strategy
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Causal model import failed: {e}")
            return False
    
    async def validate_causal_relationships(
        self,
        relationships: List[Dict[str, Any]],
        validation_data: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Validate proposed causal relationships against evidence."""
        try:
            validation_results = {
                "timestamp": datetime.utcnow().isoformat(),
                "total_relationships": len(relationships),
                "validated_relationships": [],
                "rejected_relationships": [],
                "validation_scores": {},
                "recommendations": []
            }
            
            for i, relationship in enumerate(relationships):
                cause = relationship.get("cause")
                effect = relationship.get("effect")
                proposed_strength = relationship.get("strength", 0.5)
                
                if not cause or not effect:
                    validation_results["rejected_relationships"].append({
                        "index": i,
                        "reason": "Missing cause or effect",
                        "relationship": relationship
                    })
                    continue
                
                # Validate using multiple approaches
                validation_score = 0.0
                validation_details = {}
                
                # Check if relationship exists in current graph
                if self.causal_graph.graph.has_edge(cause, effect):
                    existing_strength = self.causal_graph.graph[cause][effect].get("causal_strength", 0)
                    strength_similarity = 1.0 - abs(existing_strength - proposed_strength)
                    validation_score += strength_similarity * 0.4
                    validation_details["existing_relationship"] = True
                    validation_details["strength_match"] = strength_similarity
                else:
                    validation_details["existing_relationship"] = False
                
                # Check for conflicting relationships
                if self.causal_graph.graph.has_edge(effect, cause):
                    conflicting_strength = self.causal_graph.graph[effect][cause].get("causal_strength", 0)
                    if conflicting_strength > 0.5:
                        validation_score -= 0.3  # Penalty for reverse causation
                        validation_details["reverse_causation_conflict"] = True
                
                # Validate against external data if available
                if validation_data:
                    data_support = await self._validate_against_data(
                        cause, effect, proposed_strength, validation_data
                    )
                    validation_score += data_support * 0.6
                    validation_details["data_support"] = data_support
                
                # Check for causal pathway plausibility
                pathway_score = await self._assess_causal_pathway(cause, effect)
                validation_score += pathway_score * 0.3
                validation_details["pathway_plausibility"] = pathway_score
                
                # Determine validation result
                validation_threshold = 0.6
                if validation_score >= validation_threshold:
                    validation_results["validated_relationships"].append({
                        "index": i,
                        "cause": cause,
                        "effect": effect,
                        "proposed_strength": proposed_strength,
                        "validation_score": validation_score,
                        "details": validation_details
                    })
                else:
                    validation_results["rejected_relationships"].append({
                        "index": i,
                        "cause": cause,
                        "effect": effect,
                        "proposed_strength": proposed_strength,
                        "validation_score": validation_score,
                        "reason": "Insufficient evidence",
                        "details": validation_details
                    })
                
                validation_results["validation_scores"][f"{cause}->{effect}"] = validation_score
            
            # Generate recommendations
            if validation_results["rejected_relationships"]:
                validation_results["recommendations"].append({
                    "type": "data_collection",
                    "description": f"Collect more evidence for {len(validation_results['rejected_relationships'])} rejected relationships",
                    "priority": "medium"
                })
            
            high_confidence_relationships = [
                r for r in validation_results["validated_relationships"]
                if r["validation_score"] > 0.8
            ]
            if high_confidence_relationships:
                validation_results["recommendations"].append({
                    "type": "model_update",
                    "description": f"Consider adding {len(high_confidence_relationships)} high-confidence relationships to the causal model",
                    "priority": "high"
                })
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Causal relationship validation failed: {e}")
            return {"error": str(e)}
    
    async def _validate_against_data(
        self,
        cause: str,
        effect: str,
        proposed_strength: float,
        validation_data: List[Dict[str, Any]]
    ) -> float:
        """Validate causal relationship against empirical data."""
        try:
            # Extract time series for cause and effect variables
            cause_values = []
            effect_values = []
            
            for data_point in validation_data:
                if cause in data_point and effect in data_point:
                    cause_values.append(float(data_point[cause]))
                    effect_values.append(float(data_point[effect]))
            
            if len(cause_values) < 10:  # Need minimum data points
                return 0.5  # Neutral score for insufficient data
            
            # Calculate correlation
            correlation = np.corrcoef(cause_values, effect_values)[0, 1]
            if np.isnan(correlation):
                return 0.5
            
            # Simple validation: strong positive correlation supports causation
            correlation_support = max(0, correlation)
            
            # Check temporal ordering if timestamps available
            temporal_support = 0.5  # Default neutral
            if all("timestamp" in dp for dp in validation_data):
                # This would implement more sophisticated temporal analysis
                temporal_support = 0.7  # Placeholder
            
            # Combine evidence
            data_support = (correlation_support * 0.6) + (temporal_support * 0.4)
            return min(1.0, data_support)
            
        except Exception as e:
            logger.warning(f"Data validation failed for {cause}->{effect}: {e}")
            return 0.5
    
    async def _assess_causal_pathway(self, cause: str, effect: str) -> float:
        """Assess the plausibility of a causal pathway between entities."""
        try:
            # Check if there's a plausible path in the existing graph
            if self.causal_graph.graph.has_node(cause) and self.causal_graph.graph.has_node(effect):
                try:
                    # Find shortest path
                    path = nx.shortest_path(self.causal_graph.graph, cause, effect)
                    if len(path) <= 4:  # Reasonable path length
                        # Score based on path strength
                        path_strength = 1.0
                        for i in range(len(path) - 1):
                            edge_data = self.causal_graph.graph[path[i]][path[i+1]]
                            edge_strength = edge_data.get("causal_strength", 0.5)
                            path_strength *= edge_strength
                        
                        # Decay strength by path length
                        length_penalty = 0.9 ** (len(path) - 2)
                        return path_strength * length_penalty
                except nx.NetworkXNoPath:
                    pass
            
            # Check semantic similarity of entities (simplified)
            semantic_similarity = 0.5  # Placeholder for semantic analysis
            
            # Check domain knowledge (simplified)
            domain_plausibility = 0.6  # Placeholder for domain rules
            
            return (semantic_similarity + domain_plausibility) / 2
            
        except Exception as e:
            logger.warning(f"Pathway assessment failed for {cause}->{effect}: {e}")
            return 0.5
    
    def _convert_to_graphml(self, causal_network: Dict[str, Any]) -> Dict[str, str]:
        """Convert causal network to GraphML format."""
        # This would implement GraphML conversion
        return {
            "format": "graphml",
            "data": "<?xml version='1.0' encoding='UTF-8'?><!-- GraphML conversion not implemented -->",
            "note": "GraphML conversion placeholder"
        }
    
    def _convert_to_dot(self, causal_network: Dict[str, Any]) -> Dict[str, str]:
        """Convert causal network to DOT format for Graphviz."""
        try:
            dot_lines = ["digraph CausalGraph {"]
            dot_lines.append("  rankdir=LR;")
            dot_lines.append("  node [shape=ellipse];")
            
            # Add nodes
            for node in causal_network["nodes"]:
                node_id = node["id"]
                confidence = node.get("confidence", 1.0)
                color = "green" if confidence > 0.8 else "yellow" if confidence > 0.5 else "red"
                dot_lines.append(f'  "{node_id}" [color={color}];')
            
            # Add edges
            for edge in causal_network["edges"]:
                source = edge["source"]
                target = edge["target"]
                strength = edge.get("causal_strength", 0.0)
                style = "solid" if strength > 0.5 else "dashed"
                dot_lines.append(f'  "{source}" -> "{target}" [style={style}, label="{strength:.2f}"];')
            
            dot_lines.append("}")
            
            return {
                "format": "dot",
                "data": "\n".join(dot_lines)
            }
            
        except Exception as e:
            return {
                "format": "dot",
                "error": str(e)
            }
    
    def _convert_to_cytoscape(self, causal_network: Dict[str, Any]) -> Dict[str, Any]:
        """Convert causal network to Cytoscape.js format."""
        try:
            cytoscape_data = {
                "elements": {
                    "nodes": [],
                    "edges": []
                },
                "style": [
                    {
                        "selector": "node",
                        "style": {
                            "background-color": "#666",
                            "label": "data(id)"
                        }
                    },
                    {
                        "selector": "edge",
                        "style": {
                            "width": "mapData(strength, 0, 1, 1, 5)",
                            "line-color": "#ccc",
                            "target-arrow-color": "#ccc",
                            "target-arrow-shape": "triangle"
                        }
                    }
                ]
            }
            
            # Convert nodes
            for node in causal_network["nodes"]:
                cytoscape_data["elements"]["nodes"].append({
                    "data": {
                        "id": node["id"],
                        "confidence": node.get("confidence", 1.0),
                        "type": node.get("type", "entity")
                    }
                })
            
            # Convert edges
            for edge in causal_network["edges"]:
                cytoscape_data["elements"]["edges"].append({
                    "data": {
                        "id": f"{edge['source']}-{edge['target']}",
                        "source": edge["source"],
                        "target": edge["target"],
                        "strength": edge.get("causal_strength", 0.0),
                        "confidence": edge.get("confidence", 0.0)
                    }
                })
            
            return cytoscape_data
            
        except Exception as e:
            return {"error": str(e)}
    
    async def generate_causal_insights(
        self,
        focus_entity: Optional[str] = None,
        insight_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Generate actionable insights from the causal model."""
        try:
            insights = {
                "timestamp": datetime.utcnow().isoformat(),
                "focus_entity": focus_entity,
                "insights": [],
                "recommendations": [],
                "causal_patterns": []
            }
            
            if insight_types is None:
                insight_types = ["high_impact", "causal_chains", "feedback_loops", "intervention_points"]
            
            # High impact relationships
            if "high_impact" in insight_types:
                high_impact_edges = [
                    (source, target, data) 
                    for source, target, data in self.causal_graph.graph.edges(data=True)
                    if data.get("causal_strength", 0) > 0.8
                ]
                
                if high_impact_edges:
                    insights["insights"].append({
                        "type": "high_impact",
                        "title": f"Found {len(high_impact_edges)} high-impact causal relationships",
                        "relationships": [
                            {
                                "cause": source,
                                "effect": target,
                                "strength": data.get("causal_strength", 0),
                                "confidence": data.get("confidence", 0)
                            }
                            for source, target, data in high_impact_edges[:5]  # Top 5
                        ]
                    })
            
            # Causal chains
            if "causal_chains" in insight_types:
                chains = self._find_causal_chains(focus_entity)
                if chains:
                    insights["insights"].append({
                        "type": "causal_chains",
                        "title": f"Identified {len(chains)} causal chains",
                        "chains": chains[:3]  # Top 3 chains
                    })
            
            # Feedback loops
            if "feedback_loops" in insight_types:
                cycles = list(nx.simple_cycles(self.causal_graph.graph))
                if cycles:
                    insights["insights"].append({
                        "type": "feedback_loops",
                        "title": f"Detected {len(cycles)} feedback loops",
                        "loops": cycles[:3]  # Top 3 loops
                    })
            
            # Intervention points
            if "intervention_points" in insight_types:
                intervention_points = self._identify_intervention_points(focus_entity)
                if intervention_points:
                    insights["insights"].append({
                        "type": "intervention_points",
                        "title": f"Found {len(intervention_points)} potential intervention points",
                        "points": intervention_points[:5]  # Top 5 points
                    })
            
            return insights
            
        except Exception as e:
            logger.error(f"Causal insights generation failed: {e}")
            return {"error": str(e)}
    
    def _find_causal_chains(self, focus_entity: Optional[str] = None) -> List[Dict[str, Any]]:
        """Find significant causal chains in the graph."""
        chains = []
        
        # Find all simple paths of length 3-5
        if focus_entity and focus_entity in self.causal_graph.graph:
            # Chains starting from focus entity
            for target in self.causal_graph.graph.nodes():
                if target != focus_entity:
                    try:
                        paths = list(nx.all_simple_paths(
                            self.causal_graph.graph, 
                            focus_entity, 
                            target, 
                            cutoff=4
                        ))
                        for path in paths:
                            if len(path) >= 3:
                                chain_strength = self._calculate_chain_strength(path)
                                chains.append({
                                    "path": path,
                                    "strength": chain_strength,
                                    "length": len(path)
                                })
                    except nx.NetworkXNoPath:
                        continue
        
        # Sort by strength and return top chains
        chains.sort(key=lambda x: x["strength"], reverse=True)
        return chains[:10]
    
    def _calculate_chain_strength(self, path: List[str]) -> float:
        """Calculate the overall strength of a causal chain."""
        if len(path) < 2:
            return 0.0
        
        strength = 1.0
        for i in range(len(path) - 1):
            edge_data = self.causal_graph.graph[path[i]][path[i+1]]
            edge_strength = edge_data.get("causal_strength", 0.5)
            strength *= edge_strength
        
        # Apply length penalty
        length_penalty = 0.9 ** (len(path) - 2)
        return strength * length_penalty
    
    def _identify_intervention_points(self, focus_entity: Optional[str] = None) -> List[Dict[str, Any]]:
        """Identify high-impact intervention points in the causal graph."""
        intervention_points = []
        
        # Calculate centrality measures
        betweenness = nx.betweenness_centrality(self.causal_graph.graph)
        in_degree = dict(self.causal_graph.graph.in_degree())
        out_degree = dict(self.causal_graph.graph.out_degree())
        
        for node in self.causal_graph.graph.nodes():
            # Skip focus entity as intervention target
            if node == focus_entity:
                continue
            
            # Calculate intervention score
            intervention_score = (
                betweenness.get(node, 0) * 0.4 +
                (out_degree.get(node, 0) / max(1, max(out_degree.values()))) * 0.4 +
                (in_degree.get(node, 0) / max(1, max(in_degree.values()))) * 0.2
            )
            
            if intervention_score > 0.3:  # Threshold for significance
                intervention_points.append({
                    "entity": node,
                    "intervention_score": intervention_score,
                    "betweenness_centrality": betweenness.get(node, 0),
                    "outgoing_influences": out_degree.get(node, 0),
                    "incoming_influences": in_degree.get(node, 0)
                })
        
        # Sort by intervention score
        intervention_points.sort(key=lambda x: x["intervention_score"], reverse=True)
        return intervention_points


# Export main components
__all__ = [
    "CausalInferenceEngine", 
    "CausalGraph",
    "CausalRelationship", 
    "CausalStrength",
    "CausalDirection",
    "GrangerAnalyzer",
    "TemporalCausalAnalyzer"
]