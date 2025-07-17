"""
Dynamic Fine-Tuning Engine for Adaptive Embedding Models.

This module provides dynamic fine-tuning capabilities using local LoRA fine-tuning,
online learning with gradient descent, performance monitoring with local metrics,
and rollback mechanisms using model versioning. All processing is performed locally with zero external API calls.
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Set, Any, Tuple, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import json
import math
import hashlib
import os
import shutil
import pickle

# ML and transformer imports - all local
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModel, TrainingArguments, Trainer,
    AdapterConfig, AutoAdapterModel
)
from peft import (
    LoraConfig, get_peft_model, TaskType, PeftConfig, PeftModel
)
from sentence_transformers import SentenceTransformer, losses
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import scipy.stats as stats

import structlog
from pydantic import BaseModel, Field, ConfigDict

from ...models.memory import Memory
from ...core.cache.redis_cache import RedisCache
from ...core.utils.config import settings

logger = structlog.get_logger(__name__)


class FineTuningStrategy(str, Enum):
    """Types of fine-tuning strategies."""
    LORA = "lora"                          # LoRA parameter-efficient fine-tuning
    ADAPTER = "adapter"                    # Adapter-based fine-tuning
    FULL_FINE_TUNING = "full_fine_tuning"  # Full model fine-tuning
    ONLINE_LEARNING = "online_learning"    # Incremental online learning
    CONTINUAL_LEARNING = "continual_learning"  # Continual learning with regularization
    ENSEMBLE_FINE_TUNING = "ensemble_fine_tuning"  # Fine-tune multiple models


class LearningObjective(str, Enum):
    """Learning objectives for fine-tuning."""
    SIMILARITY_LEARNING = "similarity_learning"    # Learn better similarity representations
    CLASSIFICATION = "classification"              # Improve classification accuracy
    RETRIEVAL = "retrieval"                       # Optimize for retrieval tasks
    CLUSTERING = "clustering"                     # Better clustering performance
    DOMAIN_ADAPTATION = "domain_adaptation"      # Adapt to specific domains
    MULTI_TASK = "multi_task"                    # Multiple objectives


class PerformanceMetric(str, Enum):
    """Metrics for evaluating fine-tuning performance."""
    RETRIEVAL_ACCURACY = "retrieval_accuracy"    # Retrieval task accuracy
    SIMILARITY_CORRELATION = "similarity_correlation"  # Correlation with human similarity
    CLUSTERING_SCORE = "clustering_score"        # Clustering evaluation metrics
    CLASSIFICATION_F1 = "classification_f1"      # F1 score for classification
    PERPLEXITY = "perplexity"                    # Language modeling perplexity
    EMBEDDING_QUALITY = "embedding_quality"      # Overall embedding quality


class ModelVersionState(str, Enum):
    """States for model versions."""
    TRAINING = "training"          # Currently being trained
    VALIDATION = "validation"      # Under validation
    ACTIVE = "active"             # Currently in use
    ARCHIVED = "archived"         # Archived but available
    DEPRECATED = "deprecated"     # Marked for deletion


@dataclass
class FineTuningConfig:
    """Configuration for fine-tuning parameters."""
    strategy: FineTuningStrategy
    learning_rate: float = 2e-5
    batch_size: int = 16
    max_epochs: int = 3
    warmup_steps: int = 100
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    
    # LoRA specific
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: List[str] = field(default_factory=lambda: ["query", "value"])
    
    # Adapter specific
    adapter_reduction_factor: int = 16
    adapter_non_linearity: str = "relu"
    
    # Online learning specific
    online_learning_rate: float = 1e-4
    momentum: float = 0.9
    adaptation_rate: float = 0.1
    
    # Validation
    validation_split: float = 0.2
    early_stopping_patience: int = 3
    min_improvement: float = 0.001


@dataclass
class TrainingExample:
    """Single training example for fine-tuning."""
    text: str
    label: Optional[Union[str, int, float]] = None
    positive_examples: List[str] = field(default_factory=list)
    negative_examples: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ModelVersion:
    """Represents a version of a fine-tuned model."""
    version_id: str
    base_model_name: str
    fine_tuning_config: FineTuningConfig
    state: ModelVersionState
    created_at: datetime
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    training_examples_count: int = 0
    validation_score: float = 0.0
    model_path: Optional[str] = None
    checkpoint_paths: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FineTuningSession:
    """Represents a fine-tuning session."""
    session_id: str
    model_version: ModelVersion
    start_time: datetime
    end_time: Optional[datetime] = None
    training_examples: List[TrainingExample] = field(default_factory=list)
    validation_examples: List[TrainingExample] = field(default_factory=list)
    training_logs: List[Dict[str, Any]] = field(default_factory=list)
    final_metrics: Dict[str, float] = field(default_factory=dict)
    success: bool = False


class SimilarityDataset(Dataset):
    """Dataset for similarity learning tasks."""
    
    def __init__(
        self,
        examples: List[TrainingExample],
        tokenizer: AutoTokenizer,
        max_length: int = 512
    ):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Tokenize main text
        encoding = self.tokenizer(
            example.text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        result = {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'text': example.text
        }
        
        # Add positive/negative examples if available
        if example.positive_examples:
            pos_encoding = self.tokenizer(
                example.positive_examples[0],  # Use first positive example
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            result['positive_input_ids'] = pos_encoding['input_ids'].squeeze()
            result['positive_attention_mask'] = pos_encoding['attention_mask'].squeeze()
        
        if example.negative_examples:
            neg_encoding = self.tokenizer(
                example.negative_examples[0],  # Use first negative example
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            result['negative_input_ids'] = neg_encoding['input_ids'].squeeze()
            result['negative_attention_mask'] = neg_encoding['attention_mask'].squeeze()
        
        # Add labels if available
        if example.label is not None:
            if isinstance(example.label, str):
                result['labels'] = hash(example.label) % 1000  # Simple hash for string labels
            else:
                result['labels'] = float(example.label)
        
        result['weight'] = example.weight
        
        return result


class ContrastiveLoss(nn.Module):
    """Contrastive loss for similarity learning."""
    
    def __init__(self, margin: float = 1.0, temperature: float = 0.07):
        super().__init__()
        self.margin = margin
        self.temperature = temperature
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)
    
    def forward(
        self,
        anchor_embeddings: torch.Tensor,
        positive_embeddings: torch.Tensor,
        negative_embeddings: torch.Tensor
    ) -> torch.Tensor:
        
        # Calculate similarities
        pos_sim = self.cosine_similarity(anchor_embeddings, positive_embeddings)
        neg_sim = self.cosine_similarity(anchor_embeddings, negative_embeddings)
        
        # Contrastive loss
        loss = torch.clamp(self.margin - pos_sim + neg_sim, min=0.0)
        
        return loss.mean()


class TripletLoss(nn.Module):
    """Triplet loss for embedding learning."""
    
    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
    
    def forward(
        self,
        anchor_embeddings: torch.Tensor,
        positive_embeddings: torch.Tensor,
        negative_embeddings: torch.Tensor
    ) -> torch.Tensor:
        
        distance_positive = torch.norm(anchor_embeddings - positive_embeddings, p=2, dim=1)
        distance_negative = torch.norm(anchor_embeddings - negative_embeddings, p=2, dim=1)
        
        target = torch.ones_like(distance_positive)
        loss = self.ranking_loss(distance_negative, distance_positive, target)
        
        return loss


class OnlineLearningOptimizer:
    """
    Online learning optimizer for incremental model updates.
    
    Features:
    - Gradient-based incremental updates
    - Momentum-based optimization
    - Adaptive learning rates
    - Catastrophic forgetting prevention
    """
    
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-4,
        momentum: float = 0.9,
        adaptation_rate: float = 0.1
    ):
        self.model = model
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.adaptation_rate = adaptation_rate
        
        # Initialize momentum buffers
        self.momentum_buffers = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.momentum_buffers[name] = torch.zeros_like(param.data)
        
        # For catastrophic forgetting prevention
        self.importance_weights = {}
        self.baseline_params = {}
        self._save_baseline_params()
    
    def _save_baseline_params(self):
        """Save baseline parameters for regularization."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.baseline_params[name] = param.data.clone()
    
    async def update_model(
        self,
        loss: torch.Tensor,
        importance_weight: float = 1.0,
        regularization_strength: float = 0.1
    ):
        """Perform online model update with single example."""
        
        # Compute gradients
        loss.backward(retain_graph=True)
        
        # Update parameters with momentum and regularization
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                # Get current gradient
                grad = param.grad.data
                
                # Add regularization to prevent catastrophic forgetting
                if name in self.baseline_params:
                    reg_term = regularization_strength * (param.data - self.baseline_params[name])
                    grad += reg_term
                
                # Apply momentum
                if name in self.momentum_buffers:
                    self.momentum_buffers[name] = (
                        self.momentum * self.momentum_buffers[name] +
                        (1 - self.momentum) * grad
                    )
                    effective_grad = self.momentum_buffers[name]
                else:
                    effective_grad = grad
                
                # Update parameter
                param.data -= self.learning_rate * importance_weight * effective_grad
        
        # Clear gradients
        self.model.zero_grad()
    
    def adapt_learning_rate(self, performance_change: float):
        """Adapt learning rate based on performance."""
        
        if performance_change > 0:
            # Good performance - slightly increase learning rate
            self.learning_rate *= (1 + self.adaptation_rate * 0.1)
        else:
            # Poor performance - decrease learning rate
            self.learning_rate *= (1 - self.adaptation_rate * 0.2)
        
        # Clamp learning rate
        self.learning_rate = max(1e-6, min(1e-2, self.learning_rate))
    
    def update_importance_weights(self, task_performance: Dict[str, float]):
        """Update importance weights for different tasks."""
        
        for task_name, performance in task_performance.items():
            if task_name not in self.importance_weights:
                self.importance_weights[task_name] = 1.0
            
            # Adjust importance based on performance
            if performance > 0.8:
                self.importance_weights[task_name] *= 1.1  # Increase importance
            elif performance < 0.5:
                self.importance_weights[task_name] *= 0.9  # Decrease importance
        
        # Normalize importance weights
        total_importance = sum(self.importance_weights.values())
        if total_importance > 0:
            for task_name in self.importance_weights:
                self.importance_weights[task_name] /= total_importance


class ModelVersionManager:
    """
    Manager for model versioning and rollback capabilities.
    
    Features:
    - Model version tracking and management
    - Checkpoint saving and loading
    - Performance-based version selection
    - Automatic rollback mechanisms
    - Version cleanup and archival
    """
    
    def __init__(
        self,
        base_model_path: str,
        versions_dir: str = "./model_versions",
        max_versions: int = 10
    ):
        self.base_model_path = base_model_path
        self.versions_dir = versions_dir
        self.max_versions = max_versions
        
        # Ensure versions directory exists
        os.makedirs(versions_dir, exist_ok=True)
        
        # Track model versions
        self.versions: Dict[str, ModelVersion] = {}
        self.active_version: Optional[str] = None
        
        # Load existing versions
        self._load_existing_versions()
        
        logger.info(f"ModelVersionManager initialized with {len(self.versions)} versions")
    
    def _load_existing_versions(self):
        """Load existing model versions from disk."""
        
        try:
            versions_file = os.path.join(self.versions_dir, "versions.json")
            if os.path.exists(versions_file):
                with open(versions_file, 'r') as f:
                    versions_data = json.load(f)
                
                for version_id, version_data in versions_data.items():
                    # Reconstruct ModelVersion object
                    config_data = version_data['fine_tuning_config']
                    config = FineTuningConfig(
                        strategy=FineTuningStrategy(config_data['strategy']),
                        learning_rate=config_data.get('learning_rate', 2e-5),
                        batch_size=config_data.get('batch_size', 16),
                        max_epochs=config_data.get('max_epochs', 3)
                        # Add other config fields as needed
                    )
                    
                    version = ModelVersion(
                        version_id=version_id,
                        base_model_name=version_data['base_model_name'],
                        fine_tuning_config=config,
                        state=ModelVersionState(version_data['state']),
                        created_at=datetime.fromisoformat(version_data['created_at']),
                        performance_metrics=version_data.get('performance_metrics', {}),
                        training_examples_count=version_data.get('training_examples_count', 0),
                        validation_score=version_data.get('validation_score', 0.0),
                        model_path=version_data.get('model_path'),
                        checkpoint_paths=version_data.get('checkpoint_paths', []),
                        metadata=version_data.get('metadata', {})
                    )
                    
                    self.versions[version_id] = version
                
                # Find active version
                for version_id, version in self.versions.items():
                    if version.state == ModelVersionState.ACTIVE:
                        self.active_version = version_id
                        break
        
        except Exception as e:
            logger.warning(f"Failed to load existing versions: {e}")
    
    def _save_versions_metadata(self):
        """Save versions metadata to disk."""
        
        try:
            versions_data = {}
            for version_id, version in self.versions.items():
                versions_data[version_id] = {
                    'base_model_name': version.base_model_name,
                    'fine_tuning_config': {
                        'strategy': version.fine_tuning_config.strategy.value,
                        'learning_rate': version.fine_tuning_config.learning_rate,
                        'batch_size': version.fine_tuning_config.batch_size,
                        'max_epochs': version.fine_tuning_config.max_epochs
                        # Add other config fields as needed
                    },
                    'state': version.state.value,
                    'created_at': version.created_at.isoformat(),
                    'performance_metrics': version.performance_metrics,
                    'training_examples_count': version.training_examples_count,
                    'validation_score': version.validation_score,
                    'model_path': version.model_path,
                    'checkpoint_paths': version.checkpoint_paths,
                    'metadata': version.metadata
                }
            
            versions_file = os.path.join(self.versions_dir, "versions.json")
            with open(versions_file, 'w') as f:
                json.dump(versions_data, f, indent=2)
        
        except Exception as e:
            logger.error(f"Failed to save versions metadata: {e}")
    
    async def create_version(
        self,
        base_model_name: str,
        fine_tuning_config: FineTuningConfig,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a new model version."""
        
        version_id = f"v_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{hashlib.md5(base_model_name.encode()).hexdigest()[:8]}"
        
        version = ModelVersion(
            version_id=version_id,
            base_model_name=base_model_name,
            fine_tuning_config=fine_tuning_config,
            state=ModelVersionState.TRAINING,
            created_at=datetime.utcnow(),
            metadata=metadata or {}
        )
        
        # Create version directory
        version_dir = os.path.join(self.versions_dir, version_id)
        os.makedirs(version_dir, exist_ok=True)
        
        version.model_path = version_dir
        self.versions[version_id] = version
        
        self._save_versions_metadata()
        
        logger.info(f"Created model version {version_id}")
        return version_id
    
    async def save_checkpoint(
        self,
        version_id: str,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        epoch: int,
        metrics: Dict[str, float]
    ) -> str:
        """Save model checkpoint."""
        
        if version_id not in self.versions:
            raise ValueError(f"Version {version_id} not found")
        
        version = self.versions[version_id]
        checkpoint_name = f"checkpoint_epoch_{epoch}.pt"
        checkpoint_path = os.path.join(version.model_path, checkpoint_name)
        
        # Save checkpoint
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'metrics': metrics,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        if optimizer:
            checkpoint_data['optimizer_state_dict'] = optimizer.state_dict()
        
        torch.save(checkpoint_data, checkpoint_path)
        
        # Update version
        version.checkpoint_paths.append(checkpoint_path)
        version.performance_metrics.update(metrics)
        
        self._save_versions_metadata()
        
        logger.info(f"Saved checkpoint for version {version_id} at epoch {epoch}")
        return checkpoint_path
    
    async def load_checkpoint(
        self,
        version_id: str,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        checkpoint_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Load model checkpoint."""
        
        if version_id not in self.versions:
            raise ValueError(f"Version {version_id} not found")
        
        version = self.versions[version_id]
        
        if checkpoint_path is None:
            # Use latest checkpoint
            if not version.checkpoint_paths:
                raise ValueError(f"No checkpoints found for version {version_id}")
            checkpoint_path = version.checkpoint_paths[-1]
        
        # Load checkpoint
        checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
        
        # Load model state
        model.load_state_dict(checkpoint_data['model_state_dict'])
        
        # Load optimizer state if provided
        if optimizer and 'optimizer_state_dict' in checkpoint_data:
            optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
        
        logger.info(f"Loaded checkpoint for version {version_id} from {checkpoint_path}")
        return checkpoint_data
    
    async def activate_version(self, version_id: str):
        """Activate a specific model version."""
        
        if version_id not in self.versions:
            raise ValueError(f"Version {version_id} not found")
        
        # Deactivate current active version
        if self.active_version:
            self.versions[self.active_version].state = ModelVersionState.ARCHIVED
        
        # Activate new version
        self.versions[version_id].state = ModelVersionState.ACTIVE
        self.active_version = version_id
        
        self._save_versions_metadata()
        
        logger.info(f"Activated model version {version_id}")
    
    async def rollback_to_best_version(
        self,
        metric_name: str = "validation_score",
        min_performance: float = 0.0
    ) -> Optional[str]:
        """Rollback to the best performing version."""
        
        # Find best version based on metric
        best_version = None
        best_score = min_performance
        
        for version_id, version in self.versions.items():
            if version.state in [ModelVersionState.ACTIVE, ModelVersionState.ARCHIVED]:
                score = version.performance_metrics.get(metric_name, version.validation_score)
                if score > best_score:
                    best_score = score
                    best_version = version_id
        
        if best_version:
            await self.activate_version(best_version)
            logger.info(f"Rolled back to best version {best_version} with {metric_name}={best_score:.3f}")
            return best_version
        else:
            logger.warning("No suitable version found for rollback")
            return None
    
    async def cleanup_old_versions(self, keep_count: int = None):
        """Clean up old model versions."""
        
        keep_count = keep_count or self.max_versions
        
        # Sort versions by creation time
        sorted_versions = sorted(
            self.versions.items(),
            key=lambda x: x[1].created_at,
            reverse=True
        )
        
        # Keep the most recent versions and active version
        versions_to_keep = set()
        versions_to_keep.add(self.active_version)  # Always keep active
        
        for i, (version_id, version) in enumerate(sorted_versions):
            if i < keep_count or version.state == ModelVersionState.ACTIVE:
                versions_to_keep.add(version_id)
        
        # Remove old versions
        versions_to_remove = set(self.versions.keys()) - versions_to_keep
        
        for version_id in versions_to_remove:
            version = self.versions[version_id]
            
            # Remove model files
            if version.model_path and os.path.exists(version.model_path):
                shutil.rmtree(version.model_path)
            
            # Remove from tracking
            del self.versions[version_id]
            
            logger.info(f"Cleaned up old version {version_id}")
        
        self._save_versions_metadata()
        
        if versions_to_remove:
            logger.info(f"Cleaned up {len(versions_to_remove)} old versions")
    
    def get_version_info(self, version_id: Optional[str] = None) -> Dict[str, Any]:
        """Get information about model versions."""
        
        if version_id:
            if version_id not in self.versions:
                raise ValueError(f"Version {version_id} not found")
            
            version = self.versions[version_id]
            return {
                'version_id': version.version_id,
                'base_model_name': version.base_model_name,
                'state': version.state.value,
                'created_at': version.created_at.isoformat(),
                'performance_metrics': version.performance_metrics,
                'training_examples_count': version.training_examples_count,
                'validation_score': version.validation_score,
                'checkpoint_count': len(version.checkpoint_paths),
                'metadata': version.metadata
            }
        else:
            # Return info for all versions
            return {
                'total_versions': len(self.versions),
                'active_version': self.active_version,
                'versions': {
                    version_id: {
                        'state': version.state.value,
                        'created_at': version.created_at.isoformat(),
                        'validation_score': version.validation_score,
                        'training_examples_count': version.training_examples_count
                    }
                    for version_id, version in self.versions.items()
                }
            }


class DynamicFineTuner:
    """
    Dynamic fine-tuning engine for adaptive embedding models.
    
    Features:
    - Local LoRA fine-tuning for adaptation
    - Online learning with gradient descent
    - Performance monitoring with local metrics
    - Rollback mechanisms using model versioning
    - Multiple fine-tuning strategies
    - Continual learning capabilities
    """
    
    def __init__(
        self,
        base_model_name: str,
        fine_tuning_config: FineTuningConfig,
        model_versions_dir: str = "./model_versions",
        device: str = "cpu",
        redis_cache: Optional[RedisCache] = None
    ):
        self.base_model_name = base_model_name
        self.fine_tuning_config = fine_tuning_config
        self.device = device
        self.redis_cache = redis_cache
        
        # Core components
        self.tokenizer: Optional[AutoTokenizer] = None
        self.base_model: Optional[AutoModel] = None
        self.fine_tuned_model: Optional[nn.Module] = None
        
        # Version management
        self.version_manager = ModelVersionManager(
            base_model_name, model_versions_dir
        )
        
        # Online learning
        self.online_optimizer: Optional[OnlineLearningOptimizer] = None
        
        # Training data
        self.training_examples: List[TrainingExample] = []
        self.validation_examples: List[TrainingExample] = []
        
        # Performance tracking
        self.performance_history: List[Dict[str, Any]] = []
        self.current_session: Optional[FineTuningSession] = None
        
        # Model state
        self.is_initialized = False
        self.current_version_id: Optional[str] = None
        
        logger.info(f"DynamicFineTuner initialized for model: {base_model_name}")
    
    async def initialize(self):
        """Initialize the fine-tuning engine."""
        
        try:
            # Load tokenizer and base model
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
            self.base_model = AutoModel.from_pretrained(self.base_model_name).to(self.device)
            
            # Initialize fine-tuned model based on strategy
            await self._initialize_fine_tuned_model()
            
            # Set up online optimizer if using online learning
            if self.fine_tuning_config.strategy == FineTuningStrategy.ONLINE_LEARNING:
                self.online_optimizer = OnlineLearningOptimizer(
                    self.fine_tuned_model,
                    self.fine_tuning_config.online_learning_rate,
                    self.fine_tuning_config.momentum
                )
            
            self.is_initialized = True
            
            logger.info("DynamicFineTuner initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize DynamicFineTuner: {e}")
            raise
    
    async def _initialize_fine_tuned_model(self):
        """Initialize the fine-tuned model based on strategy."""
        
        if self.fine_tuning_config.strategy == FineTuningStrategy.LORA:
            # Use LoRA for parameter-efficient fine-tuning
            lora_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=self.fine_tuning_config.lora_r,
                lora_alpha=self.fine_tuning_config.lora_alpha,
                lora_dropout=self.fine_tuning_config.lora_dropout,
                target_modules=self.fine_tuning_config.target_modules
            )
            
            self.fine_tuned_model = get_peft_model(self.base_model, lora_config)
            
        elif self.fine_tuning_config.strategy == FineTuningStrategy.ADAPTER:
            # Use adapters
            adapter_config = AdapterConfig.load(
                "pfeiffer",
                reduction_factor=self.fine_tuning_config.adapter_reduction_factor,
                non_linearity=self.fine_tuning_config.adapter_non_linearity
            )
            
            # Convert to adapter model
            self.fine_tuned_model = AutoAdapterModel.from_pretrained(self.base_model_name)
            self.fine_tuned_model.add_adapter("fine_tuning", config=adapter_config)
            self.fine_tuned_model.train_adapter("fine_tuning")
            
        else:
            # Full fine-tuning or online learning
            self.fine_tuned_model = self.base_model
        
        self.fine_tuned_model.to(self.device)
    
    async def add_training_example(
        self,
        text: str,
        label: Optional[Union[str, int, float]] = None,
        positive_examples: Optional[List[str]] = None,
        negative_examples: Optional[List[str]] = None,
        weight: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Add a training example for fine-tuning."""
        
        example = TrainingExample(
            text=text,
            label=label,
            positive_examples=positive_examples or [],
            negative_examples=negative_examples or [],
            weight=weight,
            metadata=metadata or {}
        )
        
        self.training_examples.append(example)
        
        # Perform online learning if strategy is online
        if (self.fine_tuning_config.strategy == FineTuningStrategy.ONLINE_LEARNING and
            self.online_optimizer and len(self.training_examples) % 10 == 0):  # Update every 10 examples
            
            await self._perform_online_update([example])
    
    async def _perform_online_update(self, examples: List[TrainingExample]):
        """Perform online learning update with new examples."""
        
        if not self.online_optimizer or not examples:
            return
        
        self.fine_tuned_model.train()
        
        for example in examples:
            # Prepare input
            inputs = self.tokenizer(
                example.text,
                return_tensors='pt',
                truncation=True,
                padding=True,
                max_length=512
            ).to(self.device)
            
            # Forward pass
            outputs = self.fine_tuned_model(**inputs)
            
            # Calculate loss based on objective
            if example.positive_examples and example.negative_examples:
                # Use contrastive learning
                loss = await self._calculate_contrastive_loss(
                    example.text, example.positive_examples[0], example.negative_examples[0]
                )
            else:
                # Use simple reconstruction loss
                embeddings = outputs.last_hidden_state.mean(dim=1)
                loss = torch.norm(embeddings, p=2, dim=1).mean()
            
            # Online update
            await self.online_optimizer.update_model(
                loss, example.weight
            )
        
        logger.info(f"Performed online update with {len(examples)} examples")
    
    async def _calculate_contrastive_loss(
        self,
        anchor_text: str,
        positive_text: str,
        negative_text: str
    ) -> torch.Tensor:
        """Calculate contrastive loss for triplet of texts."""
        
        # Get embeddings for all texts
        texts = [anchor_text, positive_text, negative_text]
        inputs = self.tokenizer(
            texts,
            return_tensors='pt',
            truncation=True,
            padding=True,
            max_length=512
        ).to(self.device)
        
        outputs = self.fine_tuned_model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)  # Pool embeddings
        
        anchor_emb = embeddings[0:1]
        positive_emb = embeddings[1:2]
        negative_emb = embeddings[2:3]
        
        # Calculate contrastive loss
        contrastive_loss = ContrastiveLoss()
        loss = contrastive_loss(anchor_emb, positive_emb, negative_emb)
        
        return loss
    
    async def start_fine_tuning_session(
        self,
        validation_split: float = 0.2,
        objective: LearningObjective = LearningObjective.SIMILARITY_LEARNING
    ) -> str:
        """Start a new fine-tuning session."""
        
        if not self.is_initialized:
            await self.initialize()
        
        if len(self.training_examples) < 10:
            raise ValueError("Need at least 10 training examples for fine-tuning")
        
        # Create new model version
        version_id = await self.version_manager.create_version(
            self.base_model_name,
            self.fine_tuning_config,
            {'objective': objective.value}
        )
        
        # Split data
        if validation_split > 0:
            train_examples, val_examples = train_test_split(
                self.training_examples,
                test_size=validation_split,
                random_state=42
            )
        else:
            train_examples = self.training_examples
            val_examples = []
        
        # Create session
        session = FineTuningSession(
            session_id=f"session_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            model_version=self.version_manager.versions[version_id],
            start_time=datetime.utcnow(),
            training_examples=train_examples,
            validation_examples=val_examples
        )
        
        self.current_session = session
        self.current_version_id = version_id
        
        logger.info(f"Started fine-tuning session {session.session_id} with {len(train_examples)} training examples")
        
        return session.session_id
    
    async def run_fine_tuning(
        self,
        session_id: Optional[str] = None,
        save_checkpoints: bool = True
    ) -> Dict[str, float]:
        """Run the fine-tuning process."""
        
        if not self.current_session:
            raise ValueError("No active fine-tuning session")
        
        session = self.current_session
        
        try:
            if self.fine_tuning_config.strategy == FineTuningStrategy.LORA:
                metrics = await self._run_lora_fine_tuning(session, save_checkpoints)
            elif self.fine_tuning_config.strategy == FineTuningStrategy.FULL_FINE_TUNING:
                metrics = await self._run_full_fine_tuning(session, save_checkpoints)
            elif self.fine_tuning_config.strategy == FineTuningStrategy.ONLINE_LEARNING:
                metrics = await self._run_online_learning(session)
            else:
                raise ValueError(f"Unsupported fine-tuning strategy: {self.fine_tuning_config.strategy}")
            
            # Update session
            session.end_time = datetime.utcnow()
            session.final_metrics = metrics
            session.success = True
            
            # Update version
            if self.current_version_id:
                version = self.version_manager.versions[self.current_version_id]
                version.state = ModelVersionState.VALIDATION
                version.performance_metrics.update(metrics)
                version.validation_score = metrics.get('validation_score', 0.0)
                version.training_examples_count = len(session.training_examples)
            
            logger.info(f"Fine-tuning completed successfully: {metrics}")
            
            return metrics
            
        except Exception as e:
            session.success = False
            session.end_time = datetime.utcnow()
            
            logger.error(f"Fine-tuning failed: {e}")
            raise
    
    async def _run_lora_fine_tuning(
        self,
        session: FineTuningSession,
        save_checkpoints: bool
    ) -> Dict[str, float]:
        """Run LoRA fine-tuning."""
        
        # Prepare dataset
        train_dataset = SimilarityDataset(
            session.training_examples,
            self.tokenizer,
            max_length=512
        )
        
        val_dataset = None
        if session.validation_examples:
            val_dataset = SimilarityDataset(
                session.validation_examples,
                self.tokenizer,
                max_length=512
            )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.fine_tuning_config.batch_size,
            shuffle=True,
            num_workers=0
        )
        
        val_loader = None
        if val_dataset:
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.fine_tuning_config.batch_size,
                shuffle=False,
                num_workers=0
            )
        
        # Set up optimizer
        optimizer = optim.AdamW(
            self.fine_tuned_model.parameters(),
            lr=self.fine_tuning_config.learning_rate,
            weight_decay=self.fine_tuning_config.weight_decay
        )
        
        # Training loop
        best_val_score = 0.0
        patience_counter = 0
        
        for epoch in range(self.fine_tuning_config.max_epochs):
            # Training phase
            train_loss = await self._train_epoch(train_loader, optimizer)
            
            # Validation phase
            val_loss = 0.0
            val_score = 0.0
            if val_loader:
                val_loss, val_score = await self._validate_epoch(val_loader)
            
            # Log metrics
            epoch_metrics = {
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_score': val_score,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            session.training_logs.append(epoch_metrics)
            
            # Save checkpoint
            if save_checkpoints and self.current_version_id:
                await self.version_manager.save_checkpoint(
                    self.current_version_id,
                    self.fine_tuned_model,
                    optimizer,
                    epoch,
                    epoch_metrics
                )
            
            # Early stopping
            if val_score > best_val_score + self.fine_tuning_config.min_improvement:
                best_val_score = val_score
                patience_counter = 0
            else:
                patience_counter += 1
                
                if patience_counter >= self.fine_tuning_config.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
            
            logger.info(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_score={val_score:.4f}")
        
        # Final metrics
        final_metrics = {
            'final_train_loss': train_loss,
            'final_val_loss': val_loss,
            'validation_score': best_val_score,
            'epochs_trained': epoch + 1,
            'early_stopped': patience_counter >= self.fine_tuning_config.early_stopping_patience
        }
        
        return final_metrics
    
    async def _train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer
    ) -> float:
        """Train for one epoch."""
        
        self.fine_tuned_model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in batch.items() if k not in ['text', 'weight']}
            
            # Forward pass
            outputs = self.fine_tuned_model(**{k: v for k, v in inputs.items() if k in ['input_ids', 'attention_mask']})
            
            # Calculate loss
            if 'positive_input_ids' in inputs and 'negative_input_ids' in inputs:
                # Contrastive learning
                loss = await self._calculate_batch_contrastive_loss(outputs, inputs)
            else:
                # Simple reconstruction loss
                embeddings = outputs.last_hidden_state.mean(dim=1)
                loss = torch.norm(embeddings, p=2, dim=1).mean()
            
            # Apply weights if available
            if 'weight' in batch:
                weights = batch['weight'].to(self.device)
                loss = (loss * weights).mean()
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.fine_tuned_model.parameters(),
                self.fine_tuning_config.max_grad_norm
            )
            
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    async def _validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate for one epoch."""
        
        self.fine_tuned_model.eval()
        total_loss = 0.0
        total_score = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # Move to device
                inputs = {k: v.to(self.device) for k, v in batch.items() if k not in ['text', 'weight']}
                
                # Forward pass
                outputs = self.fine_tuned_model(**{k: v for k, v in inputs.items() if k in ['input_ids', 'attention_mask']})
                
                # Calculate loss
                if 'positive_input_ids' in inputs and 'negative_input_ids' in inputs:
                    loss = await self._calculate_batch_contrastive_loss(outputs, inputs)
                    
                    # Calculate similarity score for validation
                    anchor_emb = outputs.last_hidden_state.mean(dim=1)
                    
                    pos_outputs = self.fine_tuned_model(
                        input_ids=inputs['positive_input_ids'],
                        attention_mask=inputs['positive_attention_mask']
                    )
                    pos_emb = pos_outputs.last_hidden_state.mean(dim=1)
                    
                    neg_outputs = self.fine_tuned_model(
                        input_ids=inputs['negative_input_ids'],
                        attention_mask=inputs['negative_attention_mask']
                    )
                    neg_emb = neg_outputs.last_hidden_state.mean(dim=1)
                    
                    # Calculate ranking score (anchor should be closer to positive than negative)
                    pos_sim = torch.cosine_similarity(anchor_emb, pos_emb, dim=1)
                    neg_sim = torch.cosine_similarity(anchor_emb, neg_emb, dim=1)
                    score = (pos_sim > neg_sim).float().mean()
                    
                else:
                    embeddings = outputs.last_hidden_state.mean(dim=1)
                    loss = torch.norm(embeddings, p=2, dim=1).mean()
                    score = 1.0 / (1.0 + loss.item())  # Convert loss to score
                
                total_loss += loss.item()
                total_score += score.item() if torch.is_tensor(score) else score
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        avg_score = total_score / num_batches if num_batches > 0 else 0.0
        
        return avg_loss, avg_score
    
    async def _calculate_batch_contrastive_loss(
        self,
        anchor_outputs,
        inputs: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Calculate contrastive loss for a batch."""
        
        anchor_emb = anchor_outputs.last_hidden_state.mean(dim=1)
        
        # Get positive embeddings
        pos_outputs = self.fine_tuned_model(
            input_ids=inputs['positive_input_ids'],
            attention_mask=inputs['positive_attention_mask']
        )
        pos_emb = pos_outputs.last_hidden_state.mean(dim=1)
        
        # Get negative embeddings
        neg_outputs = self.fine_tuned_model(
            input_ids=inputs['negative_input_ids'],
            attention_mask=inputs['negative_attention_mask']
        )
        neg_emb = neg_outputs.last_hidden_state.mean(dim=1)
        
        # Calculate contrastive loss
        contrastive_loss = ContrastiveLoss()
        loss = contrastive_loss(anchor_emb, pos_emb, neg_emb)
        
        return loss
    
    async def _run_full_fine_tuning(
        self,
        session: FineTuningSession,
        save_checkpoints: bool
    ) -> Dict[str, float]:
        """Run full model fine-tuning."""
        
        # Similar to LoRA fine-tuning but with all parameters trainable
        # Implementation would be similar to _run_lora_fine_tuning
        # but without the LoRA-specific configurations
        
        return await self._run_lora_fine_tuning(session, save_checkpoints)
    
    async def _run_online_learning(self, session: FineTuningSession) -> Dict[str, float]:
        """Run online learning updates."""
        
        if not self.online_optimizer:
            raise ValueError("Online optimizer not initialized")
        
        # Process examples incrementally
        total_loss = 0.0
        num_examples = 0
        
        for example in session.training_examples:
            loss = await self._calculate_contrastive_loss(
                example.text,
                example.positive_examples[0] if example.positive_examples else example.text,
                example.negative_examples[0] if example.negative_examples else example.text
            )
            
            await self.online_optimizer.update_model(loss, example.weight)
            
            total_loss += loss.item()
            num_examples += 1
        
        avg_loss = total_loss / num_examples if num_examples > 0 else 0.0
        
        # Evaluate on validation set
        val_score = 0.0
        if session.validation_examples:
            val_score = await self._evaluate_online_model(session.validation_examples)
        
        return {
            'average_loss': avg_loss,
            'validation_score': val_score,
            'examples_processed': num_examples
        }
    
    async def _evaluate_online_model(self, validation_examples: List[TrainingExample]) -> float:
        """Evaluate online learning model."""
        
        if not validation_examples:
            return 0.0
        
        self.fine_tuned_model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for example in validation_examples:
                if example.positive_examples and example.negative_examples:
                    # Calculate similarities
                    anchor_emb = await self._get_embedding(example.text)
                    pos_emb = await self._get_embedding(example.positive_examples[0])
                    neg_emb = await self._get_embedding(example.negative_examples[0])
                    
                    # Check if anchor is closer to positive than negative
                    pos_sim = np.dot(anchor_emb, pos_emb)
                    neg_sim = np.dot(anchor_emb, neg_emb)
                    
                    if pos_sim > neg_sim:
                        correct += 1
                    total += 1
        
        return correct / total if total > 0 else 0.0
    
    async def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text using current model."""
        
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            padding=True,
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.fine_tuned_model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()[0]
        
        return embedding
    
    async def evaluate_model_performance(
        self,
        test_examples: List[TrainingExample],
        metrics: List[PerformanceMetric] = None
    ) -> Dict[str, float]:
        """Evaluate model performance on test set."""
        
        metrics = metrics or [PerformanceMetric.RETRIEVAL_ACCURACY, PerformanceMetric.SIMILARITY_CORRELATION]
        
        results = {}
        
        for metric in metrics:
            if metric == PerformanceMetric.RETRIEVAL_ACCURACY:
                score = await self._evaluate_retrieval_accuracy(test_examples)
                results['retrieval_accuracy'] = score
            
            elif metric == PerformanceMetric.SIMILARITY_CORRELATION:
                score = await self._evaluate_similarity_correlation(test_examples)
                results['similarity_correlation'] = score
        
        return results
    
    async def _evaluate_retrieval_accuracy(self, test_examples: List[TrainingExample]) -> float:
        """Evaluate retrieval accuracy."""
        
        # Implementation would depend on specific retrieval task
        # For now, use similarity ranking as proxy
        return await self._evaluate_online_model(test_examples)
    
    async def _evaluate_similarity_correlation(self, test_examples: List[TrainingExample]) -> float:
        """Evaluate correlation with human similarity judgments."""
        
        # This would require human similarity ratings
        # For now, return a placeholder based on model consistency
        
        if len(test_examples) < 10:
            return 0.5
        
        # Use consistency across similar examples as proxy
        embeddings = []
        for example in test_examples[:50]:  # Limit for efficiency
            emb = await self._get_embedding(example.text)
            embeddings.append(emb)
        
        # Calculate average pairwise similarity
        similarities = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = np.dot(embeddings[i], embeddings[j])
                similarities.append(sim)
        
        # Return standard deviation as consistency measure (lower is better)
        consistency = 1.0 - np.std(similarities)
        return max(0.0, min(1.0, consistency))
    
    async def activate_best_model(self) -> Optional[str]:
        """Activate the best performing model version."""
        
        return await self.version_manager.rollback_to_best_version()
    
    async def rollback_to_version(self, version_id: str):
        """Rollback to a specific model version."""
        
        await self.version_manager.activate_version(version_id)
        
        # Reload model with the specified version
        if version_id in self.version_manager.versions:
            version = self.version_manager.versions[version_id]
            if version.checkpoint_paths:
                # Load the latest checkpoint
                checkpoint_data = await self.version_manager.load_checkpoint(
                    version_id, self.fine_tuned_model
                )
                
                logger.info(f"Rolled back to version {version_id}")
                return checkpoint_data
        
        return None
    
    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        
        version_info = self.version_manager.get_version_info()
        
        stats = {
            'total_training_examples': len(self.training_examples),
            'total_validation_examples': len(self.validation_examples),
            'current_version_id': self.current_version_id,
            'active_version_id': self.version_manager.active_version,
            'fine_tuning_strategy': self.fine_tuning_config.strategy.value,
            'version_info': version_info,
            'performance_history': self.performance_history[-10:],  # Last 10 entries
            'is_initialized': self.is_initialized
        }
        
        if self.current_session:
            stats['current_session'] = {
                'session_id': self.current_session.session_id,
                'start_time': self.current_session.start_time.isoformat(),
                'training_examples_count': len(self.current_session.training_examples),
                'validation_examples_count': len(self.current_session.validation_examples),
                'success': self.current_session.success
            }
        
        return stats
    
    async def cleanup_resources(self):
        """Clean up resources and save state."""
        
        # Save current state
        if self.training_examples:
            # Could save training examples to disk for persistence
            pass
        
        # Clean up old model versions
        await self.version_manager.cleanup_old_versions()
        
        logger.info("DynamicFineTuner resources cleaned up")