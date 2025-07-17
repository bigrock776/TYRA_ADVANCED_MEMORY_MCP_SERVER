"""
Hyperparameter Optimization System for Local Bayesian Optimization.

This module provides comprehensive hyperparameter optimization capabilities with Optuna,
multi-objective optimization using local Pareto frontier analysis, optimization history
with local storage, and result validation with cross-validation. All processing is performed locally with zero external dependencies.
"""

import asyncio
import json
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Set, Any, Union, Callable, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict, deque
import hashlib
import threading
from pathlib import Path

# Optimization and ML imports - all local
import optuna
from optuna.samplers import TPESampler, CmaEsSampler, RandomSampler
from optuna.pruners import MedianPruner, SuccessiveHalvingPruner, HyperbandPruner
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
import joblib

import structlog
from pydantic import BaseModel, Field, ConfigDict

from .online_learning import OnlineLearningConfig, LearningTaskType, ModelType
from ...core.cache.redis_cache import RedisCache
from ...core.utils.config import settings

logger = structlog.get_logger(__name__)


class OptimizationObjective(str, Enum):
    """Types of optimization objectives."""
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


class SamplerType(str, Enum):
    """Types of Optuna samplers."""
    TPE = "tpe"                    # Tree-structured Parzen Estimator
    CMAES = "cmaes"               # Covariance Matrix Adaptation Evolution Strategy
    RANDOM = "random"             # Random sampling
    GRID = "grid"                 # Grid search


class PrunerType(str, Enum):
    """Types of Optuna pruners."""
    MEDIAN = "median"             # Median pruner
    SUCCESSIVE_HALVING = "successive_halving"  # Successive halving
    HYPERBAND = "hyperband"       # Hyperband
    NONE = "none"                 # No pruning


class OptimizationStatus(str, Enum):
    """Status of optimization studies."""
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PRUNED = "pruned"
    PAUSED = "paused"


@dataclass
class HyperparameterSpace:
    """Defines the search space for hyperparameters."""
    name: str
    param_type: str  # "float", "int", "categorical", "discrete"
    low: Optional[Union[float, int]] = None
    high: Optional[Union[float, int]] = None
    choices: Optional[List[Any]] = None
    log: bool = False
    step: Optional[Union[float, int]] = None
    
    def suggest_value(self, trial: optuna.Trial) -> Any:
        """Suggest a value using Optuna trial."""
        if self.param_type == "float":
            if self.log:
                return trial.suggest_loguniform(self.name, self.low, self.high)
            else:
                return trial.suggest_uniform(self.name, self.low, self.high)
        
        elif self.param_type == "int":
            if self.log:
                return trial.suggest_int(self.name, self.low, self.high, log=True)
            else:
                return trial.suggest_int(self.name, self.low, self.high, step=self.step)
        
        elif self.param_type == "categorical":
            return trial.suggest_categorical(self.name, self.choices)
        
        elif self.param_type == "discrete":
            return trial.suggest_discrete_uniform(self.name, self.low, self.high, self.step)
        
        else:
            raise ValueError(f"Unknown parameter type: {self.param_type}")


@dataclass
class OptimizationConfig:
    """Configuration for hyperparameter optimization."""
    study_name: str
    direction: OptimizationObjective
    sampler_type: SamplerType = SamplerType.TPE
    pruner_type: PrunerType = PrunerType.MEDIAN
    n_trials: int = 100
    timeout: Optional[int] = None  # seconds
    n_jobs: int = 1
    seed: Optional[int] = 42
    
    # Cross-validation settings
    cv_folds: int = 5
    cv_scoring: str = "accuracy"  # or "neg_mean_squared_error", "f1", etc.
    
    # Early stopping
    early_stopping_rounds: Optional[int] = 10
    min_improvement: float = 0.001
    
    # Multi-objective settings
    objectives: List[str] = field(default_factory=lambda: ["primary"])
    objective_weights: List[float] = field(default_factory=lambda: [1.0])


@dataclass
class OptimizationResult:
    """Results of hyperparameter optimization."""
    study_name: str
    best_params: Dict[str, Any]
    best_value: float
    best_values: List[float]  # For multi-objective
    n_trials: int
    optimization_history: List[Dict[str, Any]]
    pareto_front: Optional[List[Dict[str, Any]]] = None
    duration: timedelta = field(default_factory=lambda: timedelta())
    status: OptimizationStatus = OptimizationStatus.COMPLETED
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "study_name": self.study_name,
            "best_params": self.best_params,
            "best_value": self.best_value,
            "best_values": self.best_values,
            "n_trials": self.n_trials,
            "optimization_history": self.optimization_history,
            "pareto_front": self.pareto_front,
            "duration": self.duration.total_seconds(),
            "status": self.status.value
        }


class ModelObjective:
    """Objective function for model optimization."""
    
    def __init__(
        self,
        model_factory: Callable,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        config: OptimizationConfig = None,
        hyperparameter_spaces: List[HyperparameterSpace] = None
    ):
        self.model_factory = model_factory
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.config = config or OptimizationConfig("default_study", OptimizationObjective.MAXIMIZE)
        self.hyperparameter_spaces = hyperparameter_spaces or []
        
        # Performance tracking
        self.trial_history: List[Dict[str, Any]] = []
        self.best_score = float('-inf') if config.direction == OptimizationObjective.MAXIMIZE else float('inf')
        self.trials_without_improvement = 0
    
    def __call__(self, trial: optuna.Trial) -> Union[float, List[float]]:
        """Objective function called by Optuna."""
        try:
            # Suggest hyperparameters
            params = {}
            for space in self.hyperparameter_spaces:
                params[space.name] = space.suggest_value(trial)
            
            # Create model with suggested parameters
            model = self.model_factory(**params)
            
            # Evaluate model
            if self.X_val is not None and self.y_val is not None:
                # Use validation set
                score = self._evaluate_with_validation_set(model, params)
            else:
                # Use cross-validation
                score = self._evaluate_with_cross_validation(model, params)
            
            # Record trial
            trial_record = {
                "trial_number": trial.number,
                "params": params,
                "score": score,
                "timestamp": datetime.utcnow().isoformat()
            }
            self.trial_history.append(trial_record)
            
            # Check for improvement
            is_improvement = False
            if self.config.direction == OptimizationObjective.MAXIMIZE:
                if score > self.best_score + self.config.min_improvement:
                    self.best_score = score
                    is_improvement = True
            else:
                if score < self.best_score - self.config.min_improvement:
                    self.best_score = score
                    is_improvement = True
            
            # Early stopping check
            if is_improvement:
                self.trials_without_improvement = 0
            else:
                self.trials_without_improvement += 1
            
            if (self.config.early_stopping_rounds and 
                self.trials_without_improvement >= self.config.early_stopping_rounds):
                trial.study.stop()
            
            # Handle multi-objective case
            if len(self.config.objectives) > 1:
                return self._calculate_multi_objective_score(model, params, score)
            
            return score
            
        except Exception as e:
            logger.error("Error in objective function", trial_number=trial.number, error=str(e))
            # Return worst possible score
            return float('-inf') if self.config.direction == OptimizationObjective.MAXIMIZE else float('inf')
    
    def _evaluate_with_validation_set(self, model, params: Dict[str, Any]) -> float:
        """Evaluate model using validation set."""
        try:
            # Train model
            model.fit(self.X_train, self.y_train)
            
            # Make predictions
            y_pred = model.predict(self.X_val)
            
            # Calculate score based on task type
            if self.config.cv_scoring == "accuracy":
                score = accuracy_score(self.y_val, y_pred)
            elif self.config.cv_scoring == "f1":
                score = f1_score(self.y_val, y_pred, average='weighted')
            elif self.config.cv_scoring == "neg_mean_squared_error":
                score = -mean_squared_error(self.y_val, y_pred)
            elif self.config.cv_scoring == "r2":
                score = r2_score(self.y_val, y_pred)
            else:
                # Default to accuracy
                score = accuracy_score(self.y_val, y_pred)
            
            return score
            
        except Exception as e:
            logger.warning("Evaluation with validation set failed", error=str(e))
            return float('-inf') if self.config.direction == OptimizationObjective.MAXIMIZE else float('inf')
    
    def _evaluate_with_cross_validation(self, model, params: Dict[str, Any]) -> float:
        """Evaluate model using cross-validation."""
        try:
            # Determine CV strategy
            if "class" in str(type(model)).lower() and hasattr(model, "predict_proba"):
                # Classification
                cv = StratifiedKFold(n_splits=self.config.cv_folds, shuffle=True, random_state=self.config.seed)
            else:
                # Regression or other
                cv = KFold(n_splits=self.config.cv_folds, shuffle=True, random_state=self.config.seed)
            
            # Perform cross-validation
            scores = cross_val_score(
                model, self.X_train, self.y_train,
                cv=cv,
                scoring=self.config.cv_scoring,
                n_jobs=1  # Keep local for safety
            )
            
            return np.mean(scores)
            
        except Exception as e:
            logger.warning("Cross-validation failed", error=str(e))
            return float('-inf') if self.config.direction == OptimizationObjective.MAXIMIZE else float('inf')
    
    def _calculate_multi_objective_score(
        self,
        model,
        params: Dict[str, Any],
        primary_score: float
    ) -> List[float]:
        """Calculate scores for multi-objective optimization."""
        scores = [primary_score]
        
        # Add additional objectives (simplified implementation)
        for i, objective in enumerate(self.config.objectives[1:], 1):
            if objective == "model_size":
                # Prefer simpler models
                complexity = self._estimate_model_complexity(model, params)
                scores.append(-complexity)  # Minimize complexity
            elif objective == "training_time":
                # Prefer faster training (would need to measure actual time)
                estimated_time = self._estimate_training_time(model, params)
                scores.append(-estimated_time)  # Minimize time
            else:
                # Default additional score
                scores.append(primary_score * 0.9)
        
        return scores
    
    def _estimate_model_complexity(self, model, params: Dict[str, Any]) -> float:
        """Estimate model complexity for multi-objective optimization."""
        complexity = 0
        
        # Parameter-based complexity estimation
        for param_name, param_value in params.items():
            if "n_estimators" in param_name:
                complexity += param_value * 0.1
            elif "max_depth" in param_name:
                complexity += param_value * 0.05
            elif "hidden_layer_sizes" in param_name:
                if isinstance(param_value, (list, tuple)):
                    complexity += sum(param_value) * 0.001
                else:
                    complexity += param_value * 0.001
        
        return complexity
    
    def _estimate_training_time(self, model, params: Dict[str, Any]) -> float:
        """Estimate training time for multi-objective optimization."""
        # Simplified time estimation based on parameters
        time_estimate = 1.0  # Base time
        
        for param_name, param_value in params.items():
            if "n_estimators" in param_name:
                time_estimate *= (param_value / 100)
            elif "max_iter" in param_name:
                time_estimate *= (param_value / 1000)
        
        return time_estimate


class HyperparameterOptimizer:
    """Manages hyperparameter optimization studies."""
    
    def __init__(self, storage_path: str = "./optimization_studies", cache: Optional[RedisCache] = None):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.cache = cache
        
        # Active studies
        self.studies: Dict[str, optuna.Study] = {}
        self.study_configs: Dict[str, OptimizationConfig] = {}
        self.study_results: Dict[str, OptimizationResult] = {}
        
        # Threading for async operations
        self.executor = None
        
        logger.info("Hyperparameter optimizer initialized")
    
    def _create_sampler(self, sampler_type: SamplerType, seed: Optional[int] = None) -> optuna.samplers.BaseSampler:
        """Create Optuna sampler based on type."""
        if sampler_type == SamplerType.TPE:
            return TPESampler(seed=seed)
        elif sampler_type == SamplerType.CMAES:
            return CmaEsSampler(seed=seed)
        elif sampler_type == SamplerType.RANDOM:
            return RandomSampler(seed=seed)
        else:
            return TPESampler(seed=seed)
    
    def _create_pruner(self, pruner_type: PrunerType) -> Optional[optuna.pruners.BasePruner]:
        """Create Optuna pruner based on type."""
        if pruner_type == PrunerType.MEDIAN:
            return MedianPruner()
        elif pruner_type == PrunerType.SUCCESSIVE_HALVING:
            return SuccessiveHalvingPruner()
        elif pruner_type == PrunerType.HYPERBAND:
            return HyperbandPruner()
        else:
            return None
    
    async def create_study(
        self,
        config: OptimizationConfig,
        hyperparameter_spaces: List[HyperparameterSpace]
    ) -> str:
        """Create a new optimization study."""
        
        # Create Optuna study
        sampler = self._create_sampler(config.sampler_type, config.seed)
        pruner = self._create_pruner(config.pruner_type)
        
        # Handle multi-objective
        if len(config.objectives) > 1:
            directions = []
            for obj in config.objectives:
                if "error" in obj.lower() or "loss" in obj.lower() or "time" in obj.lower():
                    directions.append("minimize")
                else:
                    directions.append("maximize")
            
            study = optuna.create_study(
                study_name=config.study_name,
                directions=directions,
                sampler=sampler,
                pruner=pruner
            )
        else:
            study = optuna.create_study(
                study_name=config.study_name,
                direction=config.direction.value,
                sampler=sampler,
                pruner=pruner
            )
        
        # Store study
        self.studies[config.study_name] = study
        self.study_configs[config.study_name] = config
        
        # Cache study info
        if self.cache:
            await self.cache.set(
                f"optimization_study:{config.study_name}",
                json.dumps(asdict(config)),
                ttl=86400  # 24 hours
            )
        
        logger.info("Optimization study created", study_name=config.study_name)
        return config.study_name
    
    async def optimize_model(
        self,
        study_name: str,
        model_factory: Callable,
        X_train: np.ndarray,
        y_train: np.ndarray,
        hyperparameter_spaces: List[HyperparameterSpace],
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> OptimizationResult:
        """Optimize hyperparameters for a model."""
        
        if study_name not in self.studies:
            raise ValueError(f"Study {study_name} not found")
        
        start_time = datetime.utcnow()
        study = self.studies[study_name]
        config = self.study_configs[study_name]
        
        try:
            # Create objective function
            objective = ModelObjective(
                model_factory=model_factory,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                config=config,
                hyperparameter_spaces=hyperparameter_spaces
            )
            
            # Run optimization
            study.optimize(
                objective,
                n_trials=config.n_trials,
                timeout=config.timeout,
                n_jobs=config.n_jobs
            )
            
            # Extract results
            if len(config.objectives) > 1:
                # Multi-objective
                best_params = {}
                best_values = []
                
                if study.best_trials:
                    best_trial = study.best_trials[0]  # First Pareto optimal solution
                    best_params = best_trial.params
                    best_values = best_trial.values
                
                # Calculate Pareto front
                pareto_front = self._calculate_pareto_front(study.trials)
            else:
                # Single objective
                best_params = study.best_params
                best_values = [study.best_value]
                pareto_front = None
            
            # Create result
            result = OptimizationResult(
                study_name=study_name,
                best_params=best_params,
                best_value=best_values[0] if best_values else 0,
                best_values=best_values,
                n_trials=len(study.trials),
                optimization_history=self._extract_optimization_history(study),
                pareto_front=pareto_front,
                duration=datetime.utcnow() - start_time,
                status=OptimizationStatus.COMPLETED
            )
            
            # Store result
            self.study_results[study_name] = result
            
            # Save to disk
            await self._save_study_result(study_name, result)
            
            logger.info(
                "Hyperparameter optimization completed",
                study_name=study_name,
                n_trials=result.n_trials,
                best_value=result.best_value,
                duration=result.duration.total_seconds()
            )
            
            return result
            
        except Exception as e:
            logger.error("Hyperparameter optimization failed", study_name=study_name, error=str(e))
            
            # Return failed result
            result = OptimizationResult(
                study_name=study_name,
                best_params={},
                best_value=0,
                best_values=[],
                n_trials=len(study.trials) if hasattr(study, 'trials') else 0,
                optimization_history=[],
                duration=datetime.utcnow() - start_time,
                status=OptimizationStatus.FAILED
            )
            
            return result
    
    def _calculate_pareto_front(self, trials: List[optuna.Trial]) -> List[Dict[str, Any]]:
        """Calculate Pareto front for multi-objective optimization."""
        pareto_trials = []
        
        for trial in trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                is_dominated = False
                
                for other_trial in trials:
                    if (other_trial.state == optuna.trial.TrialState.COMPLETE and 
                        other_trial != trial):
                        
                        # Check if other_trial dominates trial
                        dominates = True
                        for i, (val1, val2) in enumerate(zip(trial.values, other_trial.values)):
                            # Assuming maximization for all objectives (adjust as needed)
                            if val1 > val2:
                                dominates = False
                                break
                        
                        if dominates:
                            is_dominated = True
                            break
                
                if not is_dominated:
                    pareto_trials.append({
                        "trial_number": trial.number,
                        "params": trial.params,
                        "values": trial.values
                    })
        
        return pareto_trials
    
    def _extract_optimization_history(self, study: optuna.Study) -> List[Dict[str, Any]]:
        """Extract optimization history from study."""
        history = []
        
        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                history.append({
                    "trial_number": trial.number,
                    "params": trial.params,
                    "value": trial.value if hasattr(trial, 'value') else None,
                    "values": trial.values if hasattr(trial, 'values') else [],
                    "datetime_start": trial.datetime_start.isoformat() if trial.datetime_start else None,
                    "datetime_complete": trial.datetime_complete.isoformat() if trial.datetime_complete else None,
                    "duration": (trial.datetime_complete - trial.datetime_start).total_seconds() if trial.datetime_complete and trial.datetime_start else None
                })
        
        return history
    
    async def _save_study_result(self, study_name: str, result: OptimizationResult):
        """Save study result to disk."""
        try:
            result_file = self.storage_path / f"{study_name}_result.json"
            
            with open(result_file, 'w') as f:
                json.dump(result.to_dict(), f, indent=2, default=str)
            
            # Also save the study object
            study_file = self.storage_path / f"{study_name}_study.pkl"
            with open(study_file, 'wb') as f:
                pickle.dump(self.studies[study_name], f)
            
        except Exception as e:
            logger.error("Failed to save study result", study_name=study_name, error=str(e))
    
    async def load_study_result(self, study_name: str) -> Optional[OptimizationResult]:
        """Load study result from disk."""
        try:
            result_file = self.storage_path / f"{study_name}_result.json"
            
            if result_file.exists():
                with open(result_file, 'r') as f:
                    result_data = json.load(f)
                
                # Reconstruct result object
                result = OptimizationResult(
                    study_name=result_data["study_name"],
                    best_params=result_data["best_params"],
                    best_value=result_data["best_value"],
                    best_values=result_data["best_values"],
                    n_trials=result_data["n_trials"],
                    optimization_history=result_data["optimization_history"],
                    pareto_front=result_data.get("pareto_front"),
                    duration=timedelta(seconds=result_data["duration"]),
                    status=OptimizationStatus(result_data["status"])
                )
                
                self.study_results[study_name] = result
                return result
            
        except Exception as e:
            logger.error("Failed to load study result", study_name=study_name, error=str(e))
        
        return None
    
    def get_study_result(self, study_name: str) -> Optional[OptimizationResult]:
        """Get study result from memory."""
        return self.study_results.get(study_name)
    
    def list_studies(self) -> List[Dict[str, Any]]:
        """List all studies."""
        studies_info = []
        
        for study_name, config in self.study_configs.items():
            result = self.study_results.get(study_name)
            
            studies_info.append({
                "study_name": study_name,
                "config": asdict(config),
                "status": result.status.value if result else "unknown",
                "n_trials": result.n_trials if result else 0,
                "best_value": result.best_value if result else None,
                "duration": result.duration.total_seconds() if result else None
            })
        
        return studies_info
    
    def suggest_hyperparameter_spaces(
        self,
        model_type: ModelType,
        task_type: LearningTaskType
    ) -> List[HyperparameterSpace]:
        """Suggest hyperparameter spaces for common model types."""
        
        spaces = []
        
        if model_type == ModelType.SGD_CLASSIFIER:
            spaces = [
                HyperparameterSpace("alpha", "float", 1e-6, 1e-1, log=True),
                HyperparameterSpace("eta0", "float", 1e-4, 1e-1, log=True),
                HyperparameterSpace("l1_ratio", "float", 0, 1),
                HyperparameterSpace("penalty", "categorical", choices=["l1", "l2", "elasticnet"])
            ]
        
        elif model_type == ModelType.SGD_REGRESSOR:
            spaces = [
                HyperparameterSpace("alpha", "float", 1e-6, 1e-1, log=True),
                HyperparameterSpace("eta0", "float", 1e-4, 1e-1, log=True),
                HyperparameterSpace("l1_ratio", "float", 0, 1),
                HyperparameterSpace("penalty", "categorical", choices=["l1", "l2", "elasticnet"])
            ]
        
        elif model_type == ModelType.RANDOM_FOREST:
            spaces = [
                HyperparameterSpace("n_estimators", "int", 10, 200),
                HyperparameterSpace("max_depth", "int", 3, 20),
                HyperparameterSpace("min_samples_split", "int", 2, 20),
                HyperparameterSpace("min_samples_leaf", "int", 1, 10),
                HyperparameterSpace("max_features", "categorical", choices=["auto", "sqrt", "log2"])
            ]
        
        elif model_type == ModelType.MLP:
            spaces = [
                HyperparameterSpace("hidden_layer_sizes", "categorical", 
                                  choices=[(50,), (100,), (50, 50), (100, 50), (100, 100)]),
                HyperparameterSpace("alpha", "float", 1e-5, 1e-1, log=True),
                HyperparameterSpace("learning_rate_init", "float", 1e-4, 1e-1, log=True),
                HyperparameterSpace("activation", "categorical", choices=["relu", "tanh", "logistic"])
            ]
        
        return spaces
    
    async def auto_optimize_model(
        self,
        model_type: ModelType,
        task_type: LearningTaskType,
        model_factory: Callable,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        n_trials: int = 50
    ) -> OptimizationResult:
        """Automatically optimize a model with suggested hyperparameter spaces."""
        
        # Create study name
        study_name = f"auto_opt_{model_type.value}_{int(datetime.utcnow().timestamp())}"
        
        # Get suggested hyperparameter spaces
        hyperparameter_spaces = self.suggest_hyperparameter_spaces(model_type, task_type)
        
        # Create optimization config
        direction = OptimizationObjective.MAXIMIZE if task_type == LearningTaskType.CLASSIFICATION else OptimizationObjective.MAXIMIZE
        scoring = "accuracy" if task_type == LearningTaskType.CLASSIFICATION else "r2"
        
        config = OptimizationConfig(
            study_name=study_name,
            direction=direction,
            n_trials=n_trials,
            cv_scoring=scoring,
            early_stopping_rounds=10
        )
        
        # Create study
        await self.create_study(config, hyperparameter_spaces)
        
        # Run optimization
        result = await self.optimize_model(
            study_name=study_name,
            model_factory=model_factory,
            X_train=X_train,
            y_train=y_train,
            hyperparameter_spaces=hyperparameter_spaces,
            X_val=X_val,
            y_val=y_val
        )
        
        return result
    
    def get_optimization_insights(self, study_name: str) -> Dict[str, Any]:
        """Get insights from optimization study."""
        result = self.study_results.get(study_name)
        if not result:
            return {}
        
        history = result.optimization_history
        if not history:
            return {}
        
        # Extract values
        values = [trial["value"] for trial in history if trial["value"] is not None]
        trials = [trial["trial_number"] for trial in history if trial["value"] is not None]
        
        if not values:
            return {}
        
        # Calculate insights
        insights = {
            "convergence_analysis": {
                "initial_value": values[0],
                "final_value": values[-1],
                "best_value": max(values) if result.best_value >= 0 else min(values),
                "improvement": (max(values) - values[0]) / abs(values[0]) if values[0] != 0 else 0,
                "convergence_trial": trials[values.index(max(values))] if result.best_value >= 0 else trials[values.index(min(values))]
            },
            "parameter_importance": self._analyze_parameter_importance(history),
            "optimization_efficiency": {
                "trials_to_best": trials[values.index(max(values))] if result.best_value >= 0 else trials[values.index(min(values))],
                "efficiency_ratio": (trials[values.index(max(values))] / len(trials)) if result.best_value >= 0 else (trials[values.index(min(values))] / len(trials))
            }
        }
        
        return insights
    
    def _analyze_parameter_importance(self, history: List[Dict[str, Any]]) -> Dict[str, float]:
        """Analyze parameter importance from optimization history."""
        if len(history) < 10:
            return {}
        
        # Simple correlation analysis between parameters and performance
        param_names = set()
        for trial in history:
            param_names.update(trial["params"].keys())
        
        importance = {}
        
        for param_name in param_names:
            param_values = []
            trial_values = []
            
            for trial in history:
                if param_name in trial["params"] and trial["value"] is not None:
                    param_values.append(trial["params"][param_name])
                    trial_values.append(trial["value"])
            
            if len(param_values) >= 5:
                # Calculate correlation (simplified)
                try:
                    # Convert categorical to numerical for correlation
                    if isinstance(param_values[0], str):
                        unique_values = list(set(param_values))
                        param_values = [unique_values.index(val) for val in param_values]
                    
                    correlation = np.corrcoef(param_values, trial_values)[0, 1]
                    importance[param_name] = abs(correlation) if not np.isnan(correlation) else 0
                except Exception:
                    importance[param_name] = 0
        
        return importance
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform optimization system health check."""
        try:
            # Create test study
            test_config = OptimizationConfig(
                study_name="health_check_test",
                direction=OptimizationObjective.MAXIMIZE,
                n_trials=5
            )
            
            test_spaces = [
                HyperparameterSpace("x", "float", -10, 10),
                HyperparameterSpace("y", "float", -10, 10)
            ]
            
            study_name = await self.create_study(test_config, test_spaces)
            
            # Test objective function (simple quadratic)
            def test_objective(trial):
                x = trial.suggest_uniform("x", -10, 10)
                y = trial.suggest_uniform("y", -10, 10)
                return -(x**2 + y**2)  # Minimize distance from origin
            
            study = self.studies[study_name]
            study.optimize(test_objective, n_trials=5)
            
            # Clean up
            del self.studies[study_name]
            del self.study_configs[study_name]
            
            return {
                "status": "healthy",
                "test_study_created": True,
                "test_optimization_completed": len(study.trials) == 5,
                "best_value_found": study.best_value is not None,
                "active_studies": len(self.studies)
            }
            
        except Exception as e:
            logger.error("Hyperparameter optimization health check failed", error=str(e))
            return {
                "status": "unhealthy",
                "error": str(e),
                "active_studies": len(self.studies)
            }