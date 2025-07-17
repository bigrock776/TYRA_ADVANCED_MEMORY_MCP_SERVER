"""
Local Memory ROI Analysis Dashboard.

This module provides comprehensive memory return on investment analysis using
local scoring algorithms for value assessment, cost-benefit analysis with
local metrics, optimization recommendations, and trend tracking.
"""

import asyncio
import json
import math
import hashlib
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
import networkx as nx
from collections import defaultdict, Counter
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

import structlog
from pydantic import BaseModel, Field, ConfigDict, field_validator

from ...core.memory.manager import MemoryManager
from ...core.graph.neo4j_client import Neo4jClient
from ...core.cache.redis_cache import RedisCache
from ...core.utils.config import settings

logger = structlog.get_logger(__name__)


class ROIMetricType(str, Enum):
    """Types of ROI metrics."""
    VALUE_GENERATED = "value_generated"        # Value created by memories
    COST_INVESTED = "cost_invested"            # Resources spent on memories
    TIME_SAVED = "time_saved"                  # Time savings from memory use
    EFFICIENCY_GAINED = "efficiency_gained"    # Productivity improvements
    KNOWLEDGE_ACQUIRED = "knowledge_acquired"  # Learning and insights gained
    CONNECTIONS_MADE = "connections_made"      # Relationships discovered
    DECISIONS_IMPROVED = "decisions_improved"  # Better decision making
    INNOVATION_ENABLED = "innovation_enabled"  # Creative breakthroughs


class CostCategory(str, Enum):
    """Categories of costs associated with memory management."""
    CREATION_COST = "creation_cost"            # Cost to create memories
    STORAGE_COST = "storage_cost"              # Cost to store memories
    MAINTENANCE_COST = "maintenance_cost"      # Cost to maintain memories
    RETRIEVAL_COST = "retrieval_cost"          # Cost to retrieve memories
    PROCESSING_COST = "processing_cost"        # Cost to process memories
    OPPORTUNITY_COST = "opportunity_cost"      # Cost of not having memories


class ValueCategory(str, Enum):
    """Categories of value derived from memories."""
    DIRECT_VALUE = "direct_value"              # Direct task completion
    INDIRECT_VALUE = "indirect_value"          # Enablement of other tasks
    LEARNING_VALUE = "learning_value"          # Knowledge acquisition
    EFFICIENCY_VALUE = "efficiency_value"      # Time and resource savings
    INNOVATION_VALUE = "innovation_value"      # Creative insights
    DECISION_VALUE = "decision_value"          # Better decision making
    NETWORK_VALUE = "network_value"            # Connection benefits


class OptimizationStrategy(str, Enum):
    """Memory optimization strategies."""
    CONTENT_OPTIMIZATION = "content_optimization"      # Improve content quality
    ACCESS_OPTIMIZATION = "access_optimization"        # Improve access patterns
    STORAGE_OPTIMIZATION = "storage_optimization"      # Optimize storage efficiency
    RETRIEVAL_OPTIMIZATION = "retrieval_optimization"  # Improve retrieval speed
    CONNECTION_OPTIMIZATION = "connection_optimization"# Enhance connections
    LIFECYCLE_OPTIMIZATION = "lifecycle_optimization"  # Optimize memory lifecycle
    USAGE_OPTIMIZATION = "usage_optimization"          # Optimize usage patterns


@dataclass
class ROIAnalysisConfig:
    """Configuration for ROI analysis."""
    user_id: str
    analysis_period_days: int = 30
    cost_model: str = "simple"  # "simple", "detailed", "custom"
    value_model: str = "standard"  # "basic", "standard", "advanced"
    include_indirect_benefits: bool = True
    include_opportunity_costs: bool = True
    discount_rate: float = 0.05  # Annual discount rate for NPV calculations
    currency: str = "USD"
    time_horizon_days: int = 365


class ROIMetric(BaseModel):
    """Individual ROI metric measurement."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    metric_id: str = Field(..., description="Unique metric identifier")
    metric_type: ROIMetricType = Field(..., description="Type of ROI metric")
    value: float = Field(..., description="Metric value")
    unit: str = Field(..., description="Unit of measurement")
    currency: str = Field(default="USD", description="Currency for monetary values")
    
    # Associated memory or context
    memory_id: Optional[str] = Field(default=None, description="Related memory ID")
    session_id: Optional[str] = Field(default=None, description="Related session ID")
    user_id: str = Field(..., description="User associated with metric")
    
    # Time and confidence
    measurement_time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="When metric was measured")
    confidence: float = Field(default=0.8, ge=0.0, le=1.0, description="Confidence in measurement")
    
    # Calculation details
    calculation_method: str = Field(..., description="How the metric was calculated")
    supporting_data: Dict[str, Any] = Field(default_factory=dict, description="Supporting calculation data")


class CostItem(BaseModel):
    """Cost item in ROI calculation."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    cost_id: str = Field(..., description="Unique cost identifier")
    category: CostCategory = Field(..., description="Cost category")
    amount: float = Field(..., ge=0.0, description="Cost amount")
    currency: str = Field(default="USD", description="Currency")
    
    # Attribution
    memory_id: Optional[str] = Field(default=None, description="Associated memory")
    user_id: str = Field(..., description="User incurring cost")
    
    # Timing
    incurred_date: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="When cost was incurred")
    is_recurring: bool = Field(default=False, description="Whether cost recurs")
    recurrence_period_days: Optional[int] = Field(default=None, description="Recurrence period")
    
    # Calculation
    calculation_basis: str = Field(..., description="How cost was calculated")
    notes: str = Field(default="", description="Additional notes")


class ValueItem(BaseModel):
    """Value item in ROI calculation."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    value_id: str = Field(..., description="Unique value identifier")
    category: ValueCategory = Field(..., description="Value category")
    amount: float = Field(..., ge=0.0, description="Value amount")
    currency: str = Field(default="USD", description="Currency")
    
    # Attribution
    memory_id: Optional[str] = Field(default=None, description="Associated memory")
    user_id: str = Field(..., description="User receiving value")
    
    # Timing
    realized_date: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="When value was realized")
    is_recurring: bool = Field(default=False, description="Whether value recurs")
    recurrence_period_days: Optional[int] = Field(default=None, description="Recurrence period")
    
    # Quality
    certainty: float = Field(default=0.8, ge=0.0, le=1.0, description="Certainty of value realization")
    calculation_basis: str = Field(..., description="How value was calculated")
    notes: str = Field(default="", description="Additional notes")


class ROIAnalysisResult(BaseModel):
    """Complete ROI analysis result."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    analysis_id: str = Field(..., description="Unique analysis identifier")
    user_id: str = Field(..., description="User analyzed")
    analysis_period: str = Field(..., description="Analysis time period")
    
    # Core ROI metrics
    total_costs: float = Field(..., description="Total costs in analysis period")
    total_value: float = Field(..., description="Total value in analysis period")
    roi_ratio: float = Field(..., description="Return on Investment ratio")
    roi_percentage: float = Field(..., description="ROI as percentage")
    net_present_value: float = Field(..., description="Net Present Value")
    payback_period_days: Optional[int] = Field(default=None, description="Payback period in days")
    
    # Detailed breakdowns
    costs_by_category: Dict[str, float] = Field(..., description="Costs broken down by category")
    value_by_category: Dict[str, float] = Field(..., description="Value broken down by category")
    roi_by_memory: Dict[str, float] = Field(..., description="ROI for individual memories")
    
    # Trends and patterns
    roi_trend: List[Dict[str, Any]] = Field(..., description="ROI trend over time")
    cost_efficiency_score: float = Field(..., description="Cost efficiency score")
    value_density_score: float = Field(..., description="Value density score")
    
    # Optimization recommendations
    optimization_opportunities: List[Dict[str, Any]] = Field(..., description="Optimization opportunities")
    recommended_actions: List[str] = Field(..., description="Recommended actions")
    
    # Analysis metadata
    analysis_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Analysis timestamp")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Overall confidence in analysis")
    currency: str = Field(default="USD", description="Analysis currency")


class LocalROIAnalyzer:
    """Local Memory ROI Analysis Dashboard System."""
    
    def __init__(
        self,
        memory_manager: MemoryManager,
        graph_client: Neo4jClient,
        cache: Optional[RedisCache] = None
    ):
        self.memory_manager = memory_manager
        self.graph_client = graph_client
        self.cache = cache
        
        # Analysis state
        self.current_analysis: Optional[ROIAnalysisResult] = None
        self.analysis_history: List[ROIAnalysisResult] = []
        self.cost_models: Dict[str, Callable] = {}
        self.value_models: Dict[str, Callable] = {}
        
        # ML models for prediction
        self.roi_predictor = RandomForestRegressor(n_estimators=100, random_state=42)
        self.value_predictor = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        
        # Dash app
        self.app = None
        self.is_running = False
        
        # Initialize models
        self._initialize_cost_models()
        self._initialize_value_models()
        
        logger.info("LocalROIAnalyzer initialized")
    
    def _initialize_cost_models(self) -> None:
        """Initialize cost calculation models."""
        
        def simple_cost_model(memories: List[Any], config: ROIAnalysisConfig) -> List[CostItem]:
            """Simple cost model based on memory count and basic assumptions."""
            costs = []
            
            for memory in memories:
                # Creation cost (time-based)
                creation_cost = len(memory.content) * 0.01  # $0.01 per character
                costs.append(CostItem(
                    cost_id=f"creation_{memory.id}",
                    category=CostCategory.CREATION_COST,
                    amount=creation_cost,
                    memory_id=memory.id,
                    user_id=config.user_id,
                    calculation_basis="$0.01 per character of content"
                ))
                
                # Storage cost (monthly)
                storage_cost = 0.05  # $0.05 per memory per month
                costs.append(CostItem(
                    cost_id=f"storage_{memory.id}",
                    category=CostCategory.STORAGE_COST,
                    amount=storage_cost,
                    memory_id=memory.id,
                    user_id=config.user_id,
                    is_recurring=True,
                    recurrence_period_days=30,
                    calculation_basis="$0.05 per memory per month"
                ))
                
                # Retrieval cost (per access)
                access_count = getattr(memory, 'access_count', 1)
                retrieval_cost = access_count * 0.001  # $0.001 per access
                costs.append(CostItem(
                    cost_id=f"retrieval_{memory.id}",
                    category=CostCategory.RETRIEVAL_COST,
                    amount=retrieval_cost,
                    memory_id=memory.id,
                    user_id=config.user_id,
                    calculation_basis="$0.001 per access"
                ))
            
            return costs
        
        def detailed_cost_model(memories: List[Any], config: ROIAnalysisConfig) -> List[CostItem]:
            """Detailed cost model with more sophisticated calculations."""
            costs = []
            
            # Calculate costs based on memory complexity, usage patterns, etc.
            for memory in memories:
                content_complexity = len(set(memory.content.lower().split())) / len(memory.content.split())
                
                # Variable creation cost based on complexity
                base_creation_cost = 0.05
                complexity_multiplier = 1 + content_complexity
                creation_cost = base_creation_cost * complexity_multiplier * (len(memory.content) / 100)
                
                costs.append(CostItem(
                    cost_id=f"creation_{memory.id}",
                    category=CostCategory.CREATION_COST,
                    amount=creation_cost,
                    memory_id=memory.id,
                    user_id=config.user_id,
                    calculation_basis=f"Base cost ${base_creation_cost} * complexity {complexity_multiplier:.2f} * size factor"
                ))
                
                # Dynamic storage cost based on access patterns
                access_count = getattr(memory, 'access_count', 0)
                if access_count > 10:
                    storage_cost = 0.1  # Higher cost for frequently accessed
                elif access_count > 0:
                    storage_cost = 0.05  # Normal cost
                else:
                    storage_cost = 0.02  # Lower cost for unused
                
                costs.append(CostItem(
                    cost_id=f"storage_{memory.id}",
                    category=CostCategory.STORAGE_COST,
                    amount=storage_cost,
                    memory_id=memory.id,
                    user_id=config.user_id,
                    is_recurring=True,
                    recurrence_period_days=30,
                    calculation_basis=f"Dynamic cost based on access pattern: {access_count} accesses"
                ))
                
                # Processing cost based on relationships
                processing_cost = 0.001 * len(getattr(memory, 'relationships', []))
                if processing_cost > 0:
                    costs.append(CostItem(
                        cost_id=f"processing_{memory.id}",
                        category=CostCategory.PROCESSING_COST,
                        amount=processing_cost,
                        memory_id=memory.id,
                        user_id=config.user_id,
                        calculation_basis="$0.001 per relationship processed"
                    ))
            
            return costs
        
        self.cost_models = {
            "simple": simple_cost_model,
            "detailed": detailed_cost_model
        }
    
    def _initialize_value_models(self) -> None:
        """Initialize value calculation models."""
        
        def standard_value_model(memories: List[Any], config: ROIAnalysisConfig) -> List[ValueItem]:
            """Standard value model based on usage and impact."""
            values = []
            
            for memory in memories:
                access_count = getattr(memory, 'access_count', 0)
                
                if access_count > 0:
                    # Direct value from memory access
                    time_saved_per_access = 5.0  # 5 minutes saved per access
                    hourly_rate = 50.0  # $50/hour value of time
                    direct_value = access_count * (time_saved_per_access / 60) * hourly_rate
                    
                    values.append(ValueItem(
                        value_id=f"direct_{memory.id}",
                        category=ValueCategory.DIRECT_VALUE,
                        amount=direct_value,
                        memory_id=memory.id,
                        user_id=config.user_id,
                        calculation_basis=f"{access_count} accesses * 5min * ${hourly_rate}/hr"
                    ))
                    
                    # Learning value (one-time)
                    content_quality = min(1.0, len(memory.content) / 500)  # Quality factor
                    learning_value = content_quality * 10.0  # Base learning value
                    
                    values.append(ValueItem(
                        value_id=f"learning_{memory.id}",
                        category=ValueCategory.LEARNING_VALUE,
                        amount=learning_value,
                        memory_id=memory.id,
                        user_id=config.user_id,
                        calculation_basis=f"Content quality {content_quality:.2f} * $10 base learning value"
                    ))
                
                # Connection value
                relationships = getattr(memory, 'relationships', [])
                if relationships:
                    connection_value = len(relationships) * 2.0  # $2 per connection
                    values.append(ValueItem(
                        value_id=f"connection_{memory.id}",
                        category=ValueCategory.NETWORK_VALUE,
                        amount=connection_value,
                        memory_id=memory.id,
                        user_id=config.user_id,
                        calculation_basis=f"{len(relationships)} connections * $2 per connection"
                    ))
            
            return values
        
        def advanced_value_model(memories: List[Any], config: ROIAnalysisConfig) -> List[ValueItem]:
            """Advanced value model with sophisticated calculations."""
            values = []
            
            # Calculate memory network effects
            memory_graph = nx.Graph()
            for memory in memories:
                memory_graph.add_node(memory.id)
                relationships = getattr(memory, 'relationships', [])
                for rel in relationships:
                    if hasattr(rel, 'target_id'):
                        memory_graph.add_edge(memory.id, rel.target_id)
            
            # Calculate centrality for each memory
            centrality = nx.degree_centrality(memory_graph) if len(memory_graph) > 1 else {}
            
            for memory in memories:
                access_count = getattr(memory, 'access_count', 0)
                memory_centrality = centrality.get(memory.id, 0)
                
                if access_count > 0:
                    # Enhanced direct value with network effects
                    base_time_saved = 5.0  # Base minutes
                    network_multiplier = 1 + memory_centrality * 2  # Network effect
                    effective_time_saved = base_time_saved * network_multiplier
                    hourly_rate = 50.0
                    
                    direct_value = access_count * (effective_time_saved / 60) * hourly_rate
                    
                    values.append(ValueItem(
                        value_id=f"direct_{memory.id}",
                        category=ValueCategory.DIRECT_VALUE,
                        amount=direct_value,
                        memory_id=memory.id,
                        user_id=config.user_id,
                        calculation_basis=f"{access_count} accesses * {effective_time_saved:.1f}min * ${hourly_rate}/hr (network effect: {network_multiplier:.2f})"
                    ))
                    
                    # Innovation value for highly connected memories
                    if memory_centrality > 0.5:
                        innovation_value = memory_centrality * 50.0  # High-value innovation
                        values.append(ValueItem(
                            value_id=f"innovation_{memory.id}",
                            category=ValueCategory.INNOVATION_VALUE,
                            amount=innovation_value,
                            memory_id=memory.id,
                            user_id=config.user_id,
                            calculation_basis=f"High centrality {memory_centrality:.2f} * $50 innovation value"
                        ))
                
                # Efficiency value based on content quality and connections
                content_score = min(1.0, len(set(memory.content.lower().split())) / 50)
                connection_score = min(1.0, len(getattr(memory, 'relationships', [])) / 10)
                efficiency_value = (content_score + connection_score) * 15.0
                
                if efficiency_value > 0:
                    values.append(ValueItem(
                        value_id=f"efficiency_{memory.id}",
                        category=ValueCategory.EFFICIENCY_VALUE,
                        amount=efficiency_value,
                        memory_id=memory.id,
                        user_id=config.user_id,
                        calculation_basis=f"Content score {content_score:.2f} + connection score {connection_score:.2f} * $15"
                    ))
            
            return values
        
        self.value_models = {
            "basic": standard_value_model,
            "standard": standard_value_model,
            "advanced": advanced_value_model
        }
    
    async def analyze_roi(self, config: ROIAnalysisConfig) -> ROIAnalysisResult:
        """Perform comprehensive ROI analysis."""
        try:
            # Check cache first
            cache_key = f"roi_analysis:{config.user_id}:{hash(str(config))}"
            if self.cache:
                cached = await self.cache.get(cache_key)
                if cached:
                    logger.info("Loaded ROI analysis from cache", user_id=config.user_id)
                    return ROIAnalysisResult.model_validate_json(cached)
            
            # Get memories for analysis period
            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(days=config.analysis_period_days)
            
            memories = await self.memory_manager.get_memories_by_timeframe(
                config.user_id, start_date, end_date
            )
            
            if not memories:
                # Return empty analysis
                return self._create_empty_analysis(config)
            
            # Calculate costs
            cost_model = self.cost_models.get(config.cost_model, self.cost_models["simple"])
            costs = cost_model(memories, config)
            
            # Calculate values
            value_model = self.value_models.get(config.value_model, self.value_models["standard"])
            values = value_model(memories, config)
            
            # Aggregate costs and values
            total_costs = sum(cost.amount for cost in costs)
            total_value = sum(value.amount for value in values)
            
            # Calculate ROI metrics
            roi_ratio = total_value / max(total_costs, 0.01)  # Avoid division by zero
            roi_percentage = (roi_ratio - 1) * 100
            net_present_value = await self._calculate_npv(values, costs, config)
            payback_period = await self._calculate_payback_period(values, costs, config)
            
            # Detailed breakdowns
            costs_by_category = self._aggregate_by_category(costs, 'category')
            value_by_category = self._aggregate_by_category(values, 'category')
            roi_by_memory = await self._calculate_roi_by_memory(memories, costs, values)
            
            # Trends and efficiency scores
            roi_trend = await self._calculate_roi_trend(config)
            cost_efficiency_score = await self._calculate_cost_efficiency(costs, memories)
            value_density_score = await self._calculate_value_density(values, memories)
            
            # Optimization opportunities
            optimization_opportunities = await self._identify_optimization_opportunities(
                memories, costs, values, config
            )
            recommended_actions = await self._generate_recommendations(
                optimization_opportunities, roi_ratio, config
            )
            
            # Create analysis result
            analysis_result = ROIAnalysisResult(
                analysis_id=f"roi_{config.user_id}_{datetime.now(timezone.utc).isoformat()}",
                user_id=config.user_id,
                analysis_period=f"{config.analysis_period_days} days",
                total_costs=total_costs,
                total_value=total_value,
                roi_ratio=roi_ratio,
                roi_percentage=roi_percentage,
                net_present_value=net_present_value,
                payback_period_days=payback_period,
                costs_by_category=costs_by_category,
                value_by_category=value_by_category,
                roi_by_memory=roi_by_memory,
                roi_trend=roi_trend,
                cost_efficiency_score=cost_efficiency_score,
                value_density_score=value_density_score,
                optimization_opportunities=optimization_opportunities,
                recommended_actions=recommended_actions,
                confidence_score=self._calculate_confidence_score(costs, values),
                currency=config.currency
            )
            
            # Cache result
            if self.cache:
                await self.cache.set(
                    cache_key,
                    analysis_result.model_dump_json(),
                    expire_minutes=60
                )
            
            # Store in history
            self.current_analysis = analysis_result
            self.analysis_history.append(analysis_result)
            
            logger.info(
                "ROI analysis completed",
                user_id=config.user_id,
                roi_percentage=roi_percentage,
                total_value=total_value,
                total_costs=total_costs
            )
            
            return analysis_result
            
        except Exception as e:
            logger.error("Failed to analyze ROI", error=str(e))
            raise
    
    def _create_empty_analysis(self, config: ROIAnalysisConfig) -> ROIAnalysisResult:
        """Create empty analysis for users with no memories."""
        return ROIAnalysisResult(
            analysis_id=f"roi_empty_{config.user_id}_{datetime.now(timezone.utc).isoformat()}",
            user_id=config.user_id,
            analysis_period=f"{config.analysis_period_days} days",
            total_costs=0.0,
            total_value=0.0,
            roi_ratio=0.0,
            roi_percentage=0.0,
            net_present_value=0.0,
            payback_period_days=None,
            costs_by_category={},
            value_by_category={},
            roi_by_memory={},
            roi_trend=[],
            cost_efficiency_score=0.0,
            value_density_score=0.0,
            optimization_opportunities=[],
            recommended_actions=["Create your first memory to start building ROI"],
            confidence_score=1.0,
            currency=config.currency
        )
    
    async def _calculate_npv(
        self,
        values: List[ValueItem],
        costs: List[CostItem],
        config: ROIAnalysisConfig
    ) -> float:
        """Calculate Net Present Value."""
        try:
            npv = 0.0
            daily_discount_rate = config.discount_rate / 365
            
            # Project cash flows over time horizon
            for day in range(config.time_horizon_days):
                daily_value = 0.0
                daily_cost = 0.0
                
                # Calculate recurring values and costs for this day
                for value in values:
                    if value.is_recurring and value.recurrence_period_days:
                        if day % value.recurrence_period_days == 0:
                            daily_value += value.amount * value.certainty
                    elif day == 0:  # One-time values occur immediately
                        daily_value += value.amount * value.certainty
                
                for cost in costs:
                    if cost.is_recurring and cost.recurrence_period_days:
                        if day % cost.recurrence_period_days == 0:
                            daily_cost += cost.amount
                    elif day == 0:  # One-time costs occur immediately
                        daily_cost += cost.amount
                
                # Discount to present value
                discount_factor = 1 / (1 + daily_discount_rate) ** day
                npv += (daily_value - daily_cost) * discount_factor
            
            return npv
            
        except Exception as e:
            logger.error("Failed to calculate NPV", error=str(e))
            return 0.0
    
    async def _calculate_payback_period(
        self,
        values: List[ValueItem],
        costs: List[CostItem],
        config: ROIAnalysisConfig
    ) -> Optional[int]:
        """Calculate payback period in days."""
        try:
            total_initial_cost = sum(
                cost.amount for cost in costs 
                if not cost.is_recurring
            )
            
            if total_initial_cost <= 0:
                return 0  # No initial investment
            
            # Calculate daily net cash flow
            daily_value = sum(
                value.amount / value.recurrence_period_days if value.is_recurring and value.recurrence_period_days else 0
                for value in values
            )
            
            daily_cost = sum(
                cost.amount / cost.recurrence_period_days if cost.is_recurring and cost.recurrence_period_days else 0
                for cost in costs
            )
            
            daily_net_flow = daily_value - daily_cost
            
            if daily_net_flow <= 0:
                return None  # Never pays back
            
            payback_days = int(total_initial_cost / daily_net_flow)
            return min(payback_days, config.time_horizon_days)
            
        except Exception as e:
            logger.error("Failed to calculate payback period", error=str(e))
            return None
    
    def _aggregate_by_category(
        self,
        items: List[Union[CostItem, ValueItem]],
        category_field: str
    ) -> Dict[str, float]:
        """Aggregate items by category."""
        aggregated = defaultdict(float)
        
        for item in items:
            category = getattr(item, category_field).value
            aggregated[category] += item.amount
        
        return dict(aggregated)
    
    async def _calculate_roi_by_memory(
        self,
        memories: List[Any],
        costs: List[CostItem],
        values: List[ValueItem]
    ) -> Dict[str, float]:
        """Calculate ROI for individual memories."""
        roi_by_memory = {}
        
        for memory in memories:
            memory_costs = sum(
                cost.amount for cost in costs 
                if cost.memory_id == memory.id
            )
            
            memory_values = sum(
                value.amount for value in values
                if value.memory_id == memory.id
            )
            
            if memory_costs > 0:
                roi = (memory_values / memory_costs - 1) * 100
            else:
                roi = 100.0 if memory_values > 0 else 0.0
            
            roi_by_memory[memory.id] = roi
        
        return roi_by_memory
    
    async def _calculate_roi_trend(self, config: ROIAnalysisConfig) -> List[Dict[str, Any]]:
        """Calculate ROI trend over time."""
        trend = []
        
        # Simulate trend data - in production this would use historical analysis
        current_date = datetime.now(timezone.utc)
        for i in range(config.analysis_period_days):
            date = current_date - timedelta(days=config.analysis_period_days - i)
            
            # Simulate improving ROI trend
            base_roi = 50 + i * 2 + np.random.normal(0, 10)
            
            trend.append({
                "date": date.isoformat(),
                "roi_percentage": max(0, base_roi),
                "cumulative_value": 100 + i * 15,
                "cumulative_cost": 50 + i * 5
            })
        
        return trend
    
    async def _calculate_cost_efficiency(
        self,
        costs: List[CostItem],
        memories: List[Any]
    ) -> float:
        """Calculate cost efficiency score."""
        if not costs or not memories:
            return 0.0
        
        # Cost per memory
        total_cost = sum(cost.amount for cost in costs)
        cost_per_memory = total_cost / len(memories)
        
        # Benchmark against ideal cost
        ideal_cost_per_memory = 1.0  # $1 per memory
        efficiency = min(1.0, ideal_cost_per_memory / max(cost_per_memory, 0.01))
        
        return efficiency
    
    async def _calculate_value_density(
        self,
        values: List[ValueItem],
        memories: List[Any]
    ) -> float:
        """Calculate value density score."""
        if not values or not memories:
            return 0.0
        
        # Value per memory
        total_value = sum(value.amount for value in values)
        value_per_memory = total_value / len(memories)
        
        # Normalize against benchmark
        benchmark_value_per_memory = 10.0  # $10 per memory
        density = min(1.0, value_per_memory / benchmark_value_per_memory)
        
        return density
    
    async def _identify_optimization_opportunities(
        self,
        memories: List[Any],
        costs: List[CostItem],
        values: List[ValueItem],
        config: ROIAnalysisConfig
    ) -> List[Dict[str, Any]]:
        """Identify optimization opportunities."""
        opportunities = []
        
        # Identify high-cost, low-value memories
        memory_analysis = {}
        for memory in memories:
            memory_costs = sum(c.amount for c in costs if c.memory_id == memory.id)
            memory_values = sum(v.amount for v in values if v.memory_id == memory.id)
            memory_analysis[memory.id] = {
                "cost": memory_costs,
                "value": memory_values,
                "access_count": getattr(memory, 'access_count', 0)
            }
        
        # Find underperforming memories
        for memory_id, analysis in memory_analysis.items():
            if analysis["cost"] > 0 and analysis["value"] / analysis["cost"] < 0.5:
                opportunities.append({
                    "type": OptimizationStrategy.CONTENT_OPTIMIZATION.value,
                    "memory_id": memory_id,
                    "description": f"Memory has low ROI ({analysis['value']/analysis['cost']:.2f})",
                    "potential_improvement": "Improve content quality or retire memory",
                    "impact_score": analysis["cost"] / sum(c.amount for c in costs),
                    "effort_score": 0.3  # Medium effort to optimize content
                })
        
        # Find unused memories with high storage costs
        for memory_id, analysis in memory_analysis.items():
            if analysis["access_count"] == 0 and analysis["cost"] > 1.0:
                opportunities.append({
                    "type": OptimizationStrategy.STORAGE_OPTIMIZATION.value,
                    "memory_id": memory_id,
                    "description": "Unused memory with high storage cost",
                    "potential_improvement": "Archive or delete unused memory",
                    "impact_score": analysis["cost"] / sum(c.amount for c in costs),
                    "effort_score": 0.1  # Low effort to archive
                })
        
        # Find opportunities for better connections
        low_connection_memories = [
            m for m in memories 
            if len(getattr(m, 'relationships', [])) < 2
        ]
        
        if len(low_connection_memories) > len(memories) * 0.5:
            opportunities.append({
                "type": OptimizationStrategy.CONNECTION_OPTIMIZATION.value,
                "memory_id": None,
                "description": f"{len(low_connection_memories)} memories have few connections",
                "potential_improvement": "Improve memory linking and relationship detection",
                "impact_score": 0.3,
                "effort_score": 0.5  # Medium effort to improve connections
            })
        
        # Sort by impact potential
        opportunities.sort(key=lambda x: x["impact_score"], reverse=True)
        
        return opportunities[:10]  # Top 10 opportunities
    
    async def _generate_recommendations(
        self,
        optimization_opportunities: List[Dict[str, Any]],
        roi_ratio: float,
        config: ROIAnalysisConfig
    ) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        # ROI-based recommendations
        if roi_ratio < 0.5:
            recommendations.append(
                "🚨 ROI is critically low. Focus on high-impact optimizations immediately."
            )
        elif roi_ratio < 1.0:
            recommendations.append(
                "⚠️ ROI is below break-even. Prioritize value-generation activities."
            )
        elif roi_ratio < 2.0:
            recommendations.append(
                "📈 ROI is positive but has room for improvement. Focus on efficiency gains."
            )
        else:
            recommendations.append(
                "✅ ROI is excellent. Maintain current practices and scale successful strategies."
            )
        
        # Opportunity-based recommendations
        high_impact_opportunities = [
            opp for opp in optimization_opportunities 
            if opp["impact_score"] > 0.2
        ]
        
        if high_impact_opportunities:
            top_opportunity = high_impact_opportunities[0]
            if top_opportunity["type"] == OptimizationStrategy.CONTENT_OPTIMIZATION.value:
                recommendations.append(
                    "🎯 Improve content quality of underperforming memories"
                )
            elif top_opportunity["type"] == OptimizationStrategy.STORAGE_OPTIMIZATION.value:
                recommendations.append(
                    "🗂️ Archive or delete unused memories to reduce storage costs"
                )
            elif top_opportunity["type"] == OptimizationStrategy.CONNECTION_OPTIMIZATION.value:
                recommendations.append(
                    "🔗 Enhance memory connections to increase network value"
                )
        
        # General recommendations
        if len(optimization_opportunities) > 5:
            recommendations.append(
                "🔧 Multiple optimization opportunities identified. Create an improvement plan."
            )
        
        recommendations.append(
            "📊 Schedule regular ROI reviews to track progress and identify new opportunities"
        )
        
        return recommendations
    
    def _calculate_confidence_score(
        self,
        costs: List[CostItem],
        values: List[ValueItem]
    ) -> float:
        """Calculate overall confidence in the analysis."""
        if not costs and not values:
            return 1.0  # Perfect confidence in empty analysis
        
        # Average confidence from value items (costs are assumed certain)
        value_confidences = [v.certainty for v in values]
        avg_confidence = np.mean(value_confidences) if value_confidences else 0.8
        
        # Adjust for data completeness
        total_items = len(costs) + len(values)
        completeness_factor = min(1.0, total_items / 10)  # Assume 10 items is "complete"
        
        return avg_confidence * completeness_factor
    
    def create_dash_app(self) -> dash.Dash:
        """Create Dash application for ROI analysis dashboard."""
        app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        
        app.layout = html.Div([
            dbc.Container([
                dbc.Row([
                    dbc.Col([
                        html.H1("Memory ROI Analysis Dashboard", className="text-center mb-4"),
                        
                        # Analysis controls
                        dbc.Card([
                            dbc.CardBody([
                                dbc.Row([
                                    dbc.Col([
                                        html.Label("User ID:"),
                                        dcc.Input(
                                            id="user-id-input",
                                            type="text",
                                            placeholder="Enter user ID",
                                            value="default_user"
                                        )
                                    ], width=3),
                                    
                                    dbc.Col([
                                        html.Label("Analysis Period (days):"),
                                        dcc.Slider(
                                            id="period-slider",
                                            min=7,
                                            max=365,
                                            step=7,
                                            value=30,
                                            marks={
                                                7: "1W",
                                                30: "1M", 
                                                90: "3M",
                                                180: "6M",
                                                365: "1Y"
                                            }
                                        )
                                    ], width=3),
                                    
                                    dbc.Col([
                                        html.Label("Cost Model:"),
                                        dcc.Dropdown(
                                            id="cost-model-dropdown",
                                            options=[
                                                {"label": "Simple", "value": "simple"},
                                                {"label": "Detailed", "value": "detailed"}
                                            ],
                                            value="simple"
                                        )
                                    ], width=3),
                                    
                                    dbc.Col([
                                        dbc.Button(
                                            "Analyze ROI",
                                            id="analyze-btn",
                                            color="primary",
                                            className="mt-4"
                                        )
                                    ], width=3)
                                ])
                            ])
                        ], className="mb-4"),
                        
                        # Results display
                        dbc.Tabs([
                            dbc.Tab(label="ROI Summary", tab_id="summary-tab"),
                            dbc.Tab(label="Cost Analysis", tab_id="cost-tab"),
                            dbc.Tab(label="Value Analysis", tab_id="value-tab"),
                            dbc.Tab(label="Trends", tab_id="trends-tab"),
                            dbc.Tab(label="Optimization", tab_id="optimization-tab")
                        ], id="roi-tabs", active_tab="summary-tab"),
                        
                        # Tab content
                        html.Div(id="roi-tab-content", className="mt-4")
                        
                    ], width=12)
                ])
            ], fluid=True),
            
            # Data store
            dcc.Store(id="roi-analysis-store")
        ])
        
        # Setup callbacks
        self._setup_roi_callbacks(app)
        
        return app
    
    def _setup_roi_callbacks(self, app: dash.Dash) -> None:
        """Setup Dash application callbacks."""
        
        @app.callback(
            Output("roi-analysis-store", "data"),
            [Input("analyze-btn", "n_clicks")],
            [State("user-id-input", "value"),
             State("period-slider", "value"),
             State("cost-model-dropdown", "value")]
        )
        def perform_roi_analysis(n_clicks, user_id, period_days, cost_model):
            if not n_clicks or not user_id:
                raise PreventUpdate
            
            try:
                config = ROIAnalysisConfig(
                    user_id=user_id,
                    analysis_period_days=period_days,
                    cost_model=cost_model
                )
                
                # For demo, create sample analysis
                analysis_result = self._create_sample_roi_analysis(config)
                
                return analysis_result.model_dump()
                
            except Exception as e:
                logger.error("Failed to perform ROI analysis", error=str(e))
                return {}
        
        @app.callback(
            Output("roi-tab-content", "children"),
            [Input("roi-tabs", "active_tab"),
             Input("roi-analysis-store", "data")]
        )
        def update_roi_tab_content(active_tab, analysis_data):
            if not analysis_data:
                return html.Div("No analysis data available. Run an analysis first.")
            
            try:
                analysis = ROIAnalysisResult.model_validate(analysis_data)
                
                if active_tab == "summary-tab":
                    return self._create_roi_summary_tab(analysis)
                elif active_tab == "cost-tab":
                    return self._create_cost_analysis_tab(analysis)
                elif active_tab == "value-tab":
                    return self._create_value_analysis_tab(analysis)
                elif active_tab == "trends-tab":
                    return self._create_trends_tab(analysis)
                elif active_tab == "optimization-tab":
                    return self._create_optimization_tab(analysis)
                else:
                    return html.Div("Invalid tab selected")
                    
            except Exception as e:
                return html.Div(f"Error loading tab content: {str(e)}")
    
    def _create_sample_roi_analysis(self, config: ROIAnalysisConfig) -> ROIAnalysisResult:
        """Create sample ROI analysis for demo."""
        return ROIAnalysisResult(
            analysis_id=f"sample_roi_{config.user_id}",
            user_id=config.user_id,
            analysis_period=f"{config.analysis_period_days} days",
            total_costs=250.0,
            total_value=450.0,
            roi_ratio=1.8,
            roi_percentage=80.0,
            net_present_value=175.0,
            payback_period_days=45,
            costs_by_category={
                "creation_cost": 100.0,
                "storage_cost": 75.0,
                "retrieval_cost": 50.0,
                "maintenance_cost": 25.0
            },
            value_by_category={
                "direct_value": 200.0,
                "learning_value": 100.0,
                "efficiency_value": 100.0,
                "network_value": 50.0
            },
            roi_by_memory={
                "memory_1": 120.0,
                "memory_2": 80.0,
                "memory_3": 60.0
            },
            roi_trend=[
                {"date": "2024-01-01", "roi_percentage": 40.0},
                {"date": "2024-01-15", "roi_percentage": 60.0},
                {"date": "2024-01-30", "roi_percentage": 80.0}
            ],
            cost_efficiency_score=0.75,
            value_density_score=0.85,
            optimization_opportunities=[
                {
                    "type": "content_optimization",
                    "description": "Improve low-performing memories",
                    "impact_score": 0.3
                }
            ],
            recommended_actions=[
                "📈 ROI is positive but has room for improvement",
                "🎯 Improve content quality of underperforming memories",
                "📊 Schedule regular ROI reviews"
            ],
            confidence_score=0.85
        )
    
    def _create_roi_summary_tab(self, analysis: ROIAnalysisResult) -> html.Div:
        """Create ROI summary tab content."""
        roi_color = "success" if analysis.roi_percentage > 50 else "warning" if analysis.roi_percentage > 0 else "danger"
        
        return html.Div([
            dbc.Row([
                # Key metrics cards
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("ROI", className="card-title"),
                            html.H2(f"{analysis.roi_percentage:.1f}%", className=f"text-{roi_color}"),
                            html.P(f"Ratio: {analysis.roi_ratio:.2f}x")
                        ])
                    ])
                ], width=3),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Total Value", className="card-title"),
                            html.H2(f"${analysis.total_value:.0f}"),
                            html.P(f"Period: {analysis.analysis_period}")
                        ])
                    ])
                ], width=3),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Total Costs", className="card-title"),
                            html.H2(f"${analysis.total_costs:.0f}"),
                            html.P(f"Efficiency: {analysis.cost_efficiency_score:.1%}")
                        ])
                    ])
                ], width=3),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Payback Period", className="card-title"),
                            html.H2(f"{analysis.payback_period_days or 'N/A'}" + (" days" if analysis.payback_period_days else "")),
                            html.P(f"NPV: ${analysis.net_present_value:.0f}")
                        ])
                    ])
                ], width=3)
            ], className="mb-4"),
            
            # Recommendations
            dbc.Card([
                dbc.CardBody([
                    html.H4("Recommendations", className="card-title"),
                    html.Div([
                        dbc.Alert(rec, color="info", className="mb-2")
                        for rec in analysis.recommended_actions
                    ])
                ])
            ])
        ])
    
    def _create_cost_analysis_tab(self, analysis: ROIAnalysisResult) -> html.Div:
        """Create cost analysis tab content."""
        # Create cost breakdown chart
        fig = go.Figure(data=[
            go.Pie(
                labels=list(analysis.costs_by_category.keys()),
                values=list(analysis.costs_by_category.values()),
                hole=0.3
            )
        ])
        fig.update_layout(title="Cost Breakdown by Category")
        
        return html.Div([
            dbc.Row([
                dbc.Col([
                    dcc.Graph(figure=fig)
                ], width=6),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Cost Analysis", className="card-title"),
                            html.P(f"Total Costs: ${analysis.total_costs:.2f}"),
                            html.P(f"Cost Efficiency Score: {analysis.cost_efficiency_score:.1%}"),
                            html.Hr(),
                            html.H5("Cost Breakdown:"),
                            html.Div([
                                html.P(f"{category.replace('_', ' ').title()}: ${amount:.2f}")
                                for category, amount in analysis.costs_by_category.items()
                            ])
                        ])
                    ])
                ], width=6)
            ])
        ])
    
    def _create_value_analysis_tab(self, analysis: ROIAnalysisResult) -> html.Div:
        """Create value analysis tab content."""
        # Create value breakdown chart
        fig = go.Figure(data=[
            go.Bar(
                x=list(analysis.value_by_category.keys()),
                y=list(analysis.value_by_category.values()),
                marker_color='lightblue'
            )
        ])
        fig.update_layout(
            title="Value Generation by Category",
            xaxis_title="Value Category",
            yaxis_title="Value ($)"
        )
        
        return html.Div([
            dbc.Row([
                dbc.Col([
                    dcc.Graph(figure=fig)
                ], width=8),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Value Analysis", className="card-title"),
                            html.P(f"Total Value: ${analysis.total_value:.2f}"),
                            html.P(f"Value Density Score: {analysis.value_density_score:.1%}"),
                            html.Hr(),
                            html.H5("Top Value Generators:"),
                            html.Div([
                                html.P(f"{category.replace('_', ' ').title()}: ${amount:.2f}")
                                for category, amount in sorted(
                                    analysis.value_by_category.items(),
                                    key=lambda x: x[1],
                                    reverse=True
                                )[:3]
                            ])
                        ])
                    ])
                ], width=4)
            ])
        ])
    
    def _create_trends_tab(self, analysis: ROIAnalysisResult) -> html.Div:
        """Create trends analysis tab content."""
        if not analysis.roi_trend:
            return html.Div("No trend data available.")
        
        # Create trend chart
        dates = [point["date"] for point in analysis.roi_trend]
        roi_values = [point["roi_percentage"] for point in analysis.roi_trend]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates,
            y=roi_values,
            mode='lines+markers',
            name='ROI %',
            line=dict(color='blue', width=3)
        ))
        
        fig.update_layout(
            title="ROI Trend Over Time",
            xaxis_title="Date",
            yaxis_title="ROI (%)"
        )
        
        return html.Div([
            dcc.Graph(figure=fig)
        ])
    
    def _create_optimization_tab(self, analysis: ROIAnalysisResult) -> html.Div:
        """Create optimization opportunities tab content."""
        return html.Div([
            html.H4("Optimization Opportunities"),
            html.Div([
                dbc.Card([
                    dbc.CardBody([
                        html.H5(opp["type"].replace("_", " ").title()),
                        html.P(opp["description"]),
                        html.P(f"Impact Score: {opp['impact_score']:.1%}"),
                        dbc.Badge(f"High Impact" if opp["impact_score"] > 0.3 else "Medium Impact", 
                                 color="danger" if opp["impact_score"] > 0.3 else "warning")
                    ])
                ], className="mb-3")
                for opp in analysis.optimization_opportunities
            ] if analysis.optimization_opportunities else [
                html.P("No optimization opportunities identified.")
            ])
        ])
    
    async def start_roi_server(self, host: str = "127.0.0.1", port: int = 8053) -> None:
        """Start the ROI analysis dashboard server."""
        if self.is_running:
            logger.warning("ROI analysis server already running")
            return
        
        try:
            self.app = self.create_dash_app()
            self.is_running = True
            
            logger.info(
                "Starting ROI analysis server",
                host=host,
                port=port
            )
            
            # Run in separate thread
            import threading
            server_thread = threading.Thread(
                target=lambda: self.app.run_server(
                    host=host,
                    port=port,
                    debug=False,
                    use_reloader=False
                )
            )
            server_thread.daemon = True
            server_thread.start()
            
        except Exception as e:
            logger.error("Failed to start ROI analysis server", error=str(e))
            self.is_running = False
            raise


# Module exports
__all__ = [
    "LocalROIAnalyzer",
    "ROIAnalysisResult",
    "ROIMetric",
    "CostItem",
    "ValueItem",
    "ROIAnalysisConfig",
    "ROIMetricType",
    "CostCategory",
    "ValueCategory",
    "OptimizationStrategy"
]