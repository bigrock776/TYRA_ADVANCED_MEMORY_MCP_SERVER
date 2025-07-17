"""
Memory Personality Profiles API endpoints.

Provides advanced personalization capabilities including user preference learning,
adaptive confidence thresholds, personalized retrieval ranking, and memory style learning.
All processing is performed locally with zero external API calls.
"""

import asyncio
import time
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel, Field

from ...core.personalization.preferences.user_preference_engine import UserPreferenceEngine, PreferenceType, UserProfile
from ...core.personalization.confidence.adaptive_confidence import AdaptiveConfidenceManager, ConfidenceContext
from ...core.cache.redis_cache import RedisCache
from ...core.utils.logger import get_logger
from ...core.utils.registry import ProviderType, get_provider

logger = get_logger(__name__)

router = APIRouter()


# Request/Response Models
class UserProfileRequest(BaseModel):
    """Request to create or update user profile."""
    
    user_id: str = Field(..., description="User identifier")
    preferences: Dict[str, Any] = Field(default={}, description="User preferences")
    interaction_history: List[Dict[str, Any]] = Field(default=[], description="User interaction history")
    context_settings: Dict[str, Any] = Field(default={}, description="Context-specific settings")


class UserProfileResponse(BaseModel):
    """Response for user profile operations."""
    
    user_id: str = Field(..., description="User identifier")
    profile_version: str = Field(..., description="Profile version")
    preferences: Dict[str, Any] = Field(..., description="Learned preferences")
    confidence_thresholds: Dict[str, float] = Field(..., description="Adaptive confidence thresholds")
    personality_traits: Dict[str, Any] = Field(..., description="Identified personality traits")
    learning_statistics: Dict[str, Any] = Field(..., description="Learning progress statistics")
    last_updated: datetime = Field(..., description="Last update timestamp")


class RecommendationRequest(BaseModel):
    """Request for personalized recommendations."""
    
    user_id: str = Field(..., description="User identifier")
    context: str = Field("general", description="Context: work, personal, research, learning, etc.")
    recommendation_type: str = Field("memory", description="Type: memory, content, action, optimization")
    limit: int = Field(10, ge=1, le=50, description="Maximum number of recommendations")
    include_explanations: bool = Field(True, description="Include recommendation explanations")
    filter_criteria: Dict[str, Any] = Field(default={}, description="Additional filtering criteria")


class RecommendationResponse(BaseModel):
    """Response for personalized recommendations."""
    
    user_id: str = Field(..., description="User identifier")
    recommendations: List[Dict[str, Any]] = Field(..., description="Personalized recommendations")
    personalization_score: float = Field(..., description="Overall personalization quality score")
    context_relevance: float = Field(..., description="Context relevance score")
    diversity_score: float = Field(..., description="Recommendation diversity score")
    explanation_summary: Dict[str, Any] = Field(..., description="Summary of recommendation reasoning")


class PreferenceLearningRequest(BaseModel):
    """Request to update preference learning."""
    
    user_id: str = Field(..., description="User identifier")
    interaction_data: List[Dict[str, Any]] = Field(..., description="New interaction data")
    feedback_data: Optional[List[Dict[str, Any]]] = Field(None, description="Explicit feedback data")
    learning_mode: str = Field("incremental", description="Learning mode: incremental, batch, adaptive")


class PreferenceLearningResponse(BaseModel):
    """Response for preference learning update."""
    
    user_id: str = Field(..., description="User identifier")
    learning_summary: Dict[str, Any] = Field(..., description="Learning progress summary")
    preference_changes: Dict[str, Any] = Field(..., description="Detected preference changes")
    confidence_adjustments: Dict[str, Any] = Field(..., description="Confidence threshold adjustments")
    model_performance: Dict[str, Any] = Field(..., description="Model performance metrics")


class ConfidenceOptimizationRequest(BaseModel):
    """Request for confidence threshold optimization."""
    
    user_id: str = Field(..., description="User identifier")
    context_type: str = Field("general", description="Context type for optimization")
    optimization_goal: str = Field("balanced", description="Goal: precision, recall, balanced, f1")
    historical_data_days: int = Field(30, ge=7, le=365, description="Historical data window")
    include_cross_validation: bool = Field(True, description="Include cross-validation")


class ConfidenceOptimizationResponse(BaseModel):
    """Response for confidence threshold optimization."""
    
    user_id: str = Field(..., description="User identifier")
    optimized_thresholds: Dict[str, float] = Field(..., description="Optimized confidence thresholds")
    performance_improvement: Dict[str, float] = Field(..., description="Performance improvement metrics")
    optimization_details: Dict[str, Any] = Field(..., description="Optimization process details")
    validation_results: Optional[Dict[str, Any]] = Field(None, description="Cross-validation results")


class PersonalityAnalysisRequest(BaseModel):
    """Request for personality analysis."""
    
    user_id: str = Field(..., description="User identifier")
    analysis_depth: str = Field("standard", description="Analysis depth: quick, standard, comprehensive")
    include_evolution: bool = Field(True, description="Include personality evolution tracking")
    time_window_days: int = Field(90, ge=30, le=365, description="Analysis time window")


class PersonalityAnalysisResponse(BaseModel):
    """Response for personality analysis."""
    
    user_id: str = Field(..., description="User identifier")
    personality_profile: Dict[str, Any] = Field(..., description="Comprehensive personality profile")
    memory_style: Dict[str, Any] = Field(..., description="Memory interaction style analysis")
    behavioral_patterns: Dict[str, Any] = Field(..., description="Identified behavioral patterns")
    evolution_trends: Optional[Dict[str, Any]] = Field(None, description="Personality evolution trends")
    insights: List[Dict[str, Any]] = Field(..., description="Personality insights and recommendations")


class PersonalizationMetricsResponse(BaseModel):
    """Response for personalization system metrics."""
    
    timestamp: datetime = Field(..., description="Metrics timestamp")
    preference_engine: Dict[str, Any] = Field(..., description="Preference engine metrics")
    confidence_manager: Dict[str, Any] = Field(..., description="Confidence manager metrics")
    recommendation_system: Dict[str, Any] = Field(..., description="Recommendation system metrics")
    personalization_quality: Dict[str, Any] = Field(..., description="Overall personalization quality metrics")


# Dependencies
async def get_preference_engine() -> UserPreferenceEngine:
    """Get user preference engine instance."""
    try:
        memory_manager = get_provider(ProviderType.MEMORY_MANAGER, "default")
        cache = get_provider(ProviderType.CACHE, "default")
        return UserPreferenceEngine(memory_manager=memory_manager, cache=cache)
    except Exception as e:
        logger.error(f"Failed to get preference engine: {e}")
        raise HTTPException(status_code=500, detail="Preference engine unavailable")


async def get_confidence_manager() -> AdaptiveConfidenceManager:
    """Get adaptive confidence manager instance."""
    try:
        cache = get_provider(ProviderType.CACHE, "default")
        return AdaptiveConfidenceManager(cache=cache)
    except Exception as e:
        logger.error(f"Failed to get confidence manager: {e}")
        raise HTTPException(status_code=500, detail="Confidence manager unavailable")


# API Endpoints

@router.post("/profile", response_model=UserProfileResponse)
async def create_or_update_profile(
    request: UserProfileRequest,
    background_tasks: BackgroundTasks,
    preference_engine: UserPreferenceEngine = Depends(get_preference_engine),
    confidence_manager: AdaptiveConfidenceManager = Depends(get_confidence_manager)
):
    """
    Create or update user personality profile.
    
    Initializes or updates user profile with preferences, interaction history,
    and context settings for personalized experiences.
    """
    try:
        # Create or update user profile
        profile = await preference_engine.create_or_update_profile(
            user_id=request.user_id,
            preferences=request.preferences,
            interaction_history=request.interaction_history,
            context_settings=request.context_settings
        )
        
        # Initialize adaptive confidence thresholds
        confidence_thresholds = await confidence_manager.initialize_user_thresholds(
            user_id=request.user_id,
            initial_preferences=request.preferences
        )
        
        # Start background learning process
        if request.interaction_history:
            background_tasks.add_task(
                _background_preference_learning,
                preference_engine,
                request.user_id,
                request.interaction_history
            )
        
        return UserProfileResponse(
            user_id=request.user_id,
            profile_version=profile.version,
            preferences=profile.preferences,
            confidence_thresholds=confidence_thresholds,
            personality_traits=profile.personality_traits,
            learning_statistics=profile.learning_statistics,
            last_updated=profile.last_updated
        )
        
    except Exception as e:
        logger.error(f"Profile creation/update failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/profile/{user_id}", response_model=UserProfileResponse)
async def get_user_profile(
    user_id: str,
    include_detailed_stats: bool = Query(False, description="Include detailed learning statistics"),
    preference_engine: UserPreferenceEngine = Depends(get_preference_engine),
    confidence_manager: AdaptiveConfidenceManager = Depends(get_confidence_manager)
):
    """
    Get user personality profile.
    
    Returns comprehensive user profile including preferences, confidence thresholds,
    and personality traits with optional detailed statistics.
    """
    try:
        # Get user profile
        profile = await preference_engine.get_user_profile(user_id)
        if not profile:
            raise HTTPException(status_code=404, detail="User profile not found")
        
        # Get confidence thresholds
        confidence_thresholds = await confidence_manager.get_user_thresholds(user_id)
        
        # Get detailed statistics if requested
        learning_statistics = profile.learning_statistics
        if include_detailed_stats:
            detailed_stats = await preference_engine.get_detailed_learning_statistics(user_id)
            learning_statistics.update(detailed_stats)
        
        return UserProfileResponse(
            user_id=user_id,
            profile_version=profile.version,
            preferences=profile.preferences,
            confidence_thresholds=confidence_thresholds,
            personality_traits=profile.personality_traits,
            learning_statistics=learning_statistics,
            last_updated=profile.last_updated
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get user profile: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/recommendations", response_model=RecommendationResponse)
async def get_personalized_recommendations(
    request: RecommendationRequest,
    preference_engine: UserPreferenceEngine = Depends(get_preference_engine),
    confidence_manager: AdaptiveConfidenceManager = Depends(get_confidence_manager)
):
    """
    Get personalized recommendations for user.
    
    Generates context-aware recommendations based on user preferences,
    personality traits, and adaptive confidence thresholds.
    """
    try:
        # Create confidence context
        confidence_context = ConfidenceContext(
            context_type=request.context,
            user_id=request.user_id,
            recommendation_type=request.recommendation_type
        )
        
        # Get adaptive confidence thresholds
        confidence_thresholds = await confidence_manager.get_contextual_thresholds(
            user_id=request.user_id,
            context=confidence_context
        )
        
        # Generate personalized recommendations
        recommendations = await preference_engine.generate_recommendations(
            user_id=request.user_id,
            context=request.context,
            recommendation_type=request.recommendation_type,
            limit=request.limit,
            confidence_thresholds=confidence_thresholds,
            filter_criteria=request.filter_criteria,
            include_explanations=request.include_explanations
        )
        
        # Calculate recommendation quality scores
        personalization_score = await preference_engine.calculate_personalization_score(
            user_id=request.user_id,
            recommendations=recommendations
        )
        
        context_relevance = await preference_engine.calculate_context_relevance(
            recommendations=recommendations,
            context=request.context
        )
        
        diversity_score = await preference_engine.calculate_diversity_score(
            recommendations=recommendations
        )
        
        # Generate explanation summary
        explanation_summary = {}
        if request.include_explanations:
            explanation_summary = await preference_engine.generate_explanation_summary(
                recommendations=recommendations,
                user_id=request.user_id
            )
        
        return RecommendationResponse(
            user_id=request.user_id,
            recommendations=recommendations,
            personalization_score=personalization_score,
            context_relevance=context_relevance,
            diversity_score=diversity_score,
            explanation_summary=explanation_summary
        )
        
    except Exception as e:
        logger.error(f"Recommendation generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/learn", response_model=PreferenceLearningResponse)
async def update_preference_learning(
    request: PreferenceLearningRequest,
    background_tasks: BackgroundTasks,
    preference_engine: UserPreferenceEngine = Depends(get_preference_engine),
    confidence_manager: AdaptiveConfidenceManager = Depends(get_confidence_manager)
):
    """
    Update preference learning with new interaction data.
    
    Incorporates new user interactions and feedback to continuously
    improve personalization and adapt to changing preferences.
    """
    try:
        # Update preference learning
        learning_result = await preference_engine.update_learning(
            user_id=request.user_id,
            interaction_data=request.interaction_data,
            feedback_data=request.feedback_data,
            learning_mode=request.learning_mode
        )
        
        # Update confidence thresholds based on new learning
        confidence_adjustments = await confidence_manager.adapt_thresholds(
            user_id=request.user_id,
            interaction_data=request.interaction_data,
            performance_feedback=learning_result.get("performance_metrics", {})
        )
        
        # Schedule background model optimization if significant changes detected
        if learning_result.get("significant_changes", False):
            background_tasks.add_task(
                _background_model_optimization,
                preference_engine,
                request.user_id
            )
        
        return PreferenceLearningResponse(
            user_id=request.user_id,
            learning_summary=learning_result.get("learning_summary", {}),
            preference_changes=learning_result.get("preference_changes", {}),
            confidence_adjustments=confidence_adjustments,
            model_performance=learning_result.get("performance_metrics", {})
        )
        
    except Exception as e:
        logger.error(f"Preference learning update failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/confidence/optimize", response_model=ConfidenceOptimizationResponse)
async def optimize_confidence_thresholds(
    request: ConfidenceOptimizationRequest,
    confidence_manager: AdaptiveConfidenceManager = Depends(get_confidence_manager)
):
    """
    Optimize confidence thresholds for user.
    
    Performs systematic optimization of confidence thresholds based on
    historical performance and specified optimization goals.
    """
    try:
        # Create confidence context
        context = ConfidenceContext(
            context_type=request.context_type,
            user_id=request.user_id,
            optimization_goal=request.optimization_goal
        )
        
        # Perform threshold optimization
        optimization_result = await confidence_manager.optimize_thresholds(
            user_id=request.user_id,
            context=context,
            historical_data_days=request.historical_data_days,
            include_cross_validation=request.include_cross_validation
        )
        
        return ConfidenceOptimizationResponse(
            user_id=request.user_id,
            optimized_thresholds=optimization_result["optimized_thresholds"],
            performance_improvement=optimization_result["performance_improvement"],
            optimization_details=optimization_result["optimization_details"],
            validation_results=optimization_result.get("validation_results") if request.include_cross_validation else None
        )
        
    except Exception as e:
        logger.error(f"Confidence optimization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/personality/analyze", response_model=PersonalityAnalysisResponse)
async def analyze_personality(
    request: PersonalityAnalysisRequest,
    preference_engine: UserPreferenceEngine = Depends(get_preference_engine)
):
    """
    Perform comprehensive personality analysis.
    
    Analyzes user behavior patterns, memory interaction styles, and
    personality traits with optional evolution tracking.
    """
    try:
        # Perform personality analysis
        analysis_result = await preference_engine.analyze_personality(
            user_id=request.user_id,
            analysis_depth=request.analysis_depth,
            time_window_days=request.time_window_days,
            include_evolution=request.include_evolution
        )
        
        # Generate insights and recommendations
        insights = await preference_engine.generate_personality_insights(
            user_id=request.user_id,
            personality_profile=analysis_result["personality_profile"],
            behavioral_patterns=analysis_result["behavioral_patterns"]
        )
        
        return PersonalityAnalysisResponse(
            user_id=request.user_id,
            personality_profile=analysis_result["personality_profile"],
            memory_style=analysis_result["memory_style"],
            behavioral_patterns=analysis_result["behavioral_patterns"],
            evolution_trends=analysis_result.get("evolution_trends") if request.include_evolution else None,
            insights=insights
        )
        
    except Exception as e:
        logger.error(f"Personality analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/users/{user_id}/similarity")
async def find_similar_users(
    user_id: str,
    similarity_threshold: float = Query(0.7, ge=0.0, le=1.0, description="Minimum similarity threshold"),
    limit: int = Query(10, ge=1, le=50, description="Maximum number of similar users"),
    include_details: bool = Query(False, description="Include detailed similarity breakdown"),
    preference_engine: UserPreferenceEngine = Depends(get_preference_engine)
):
    """
    Find users with similar preferences and personality traits.
    
    Identifies users with similar behavior patterns for collaborative
    filtering and community-based recommendations.
    """
    try:
        # Find similar users
        similar_users = await preference_engine.find_similar_users(
            user_id=user_id,
            similarity_threshold=similarity_threshold,
            limit=limit,
            include_details=include_details
        )
        
        return {
            "user_id": user_id,
            "similar_users": similar_users,
            "similarity_threshold": similarity_threshold,
            "total_found": len(similar_users),
            "analysis_timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Similar user finding failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/preferences/evolution/{user_id}")
async def get_preference_evolution(
    user_id: str,
    time_range_days: int = Query(90, ge=30, le=365, description="Time range for evolution analysis"),
    granularity: str = Query("weekly", description="Evolution granularity: daily, weekly, monthly"),
    preference_types: Optional[List[str]] = Query(None, description="Specific preference types to analyze"),
    preference_engine: UserPreferenceEngine = Depends(get_preference_engine)
):
    """
    Get user preference evolution over time.
    
    Tracks how user preferences have changed over time with
    detailed evolution patterns and trend analysis.
    """
    try:
        # Get preference evolution
        evolution_data = await preference_engine.get_preference_evolution(
            user_id=user_id,
            time_range_days=time_range_days,
            granularity=granularity,
            preference_types=preference_types
        )
        
        return {
            "user_id": user_id,
            "time_range_days": time_range_days,
            "granularity": granularity,
            "evolution_data": evolution_data,
            "preference_types_analyzed": preference_types or "all",
            "analysis_timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Preference evolution analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/feedback")
async def submit_recommendation_feedback(
    user_id: str = Query(..., description="User identifier"),
    recommendation_id: str = Query(..., description="Recommendation identifier"),
    feedback_type: str = Query(..., description="Feedback type: positive, negative, neutral"),
    feedback_details: Dict[str, Any] = {},
    preference_engine: UserPreferenceEngine = Depends(get_preference_engine),
    confidence_manager: AdaptiveConfidenceManager = Depends(get_confidence_manager)
):
    """
    Submit feedback on personalized recommendations.
    
    Collects user feedback to improve recommendation quality
    and adapt personalization algorithms.
    """
    try:
        # Submit feedback to preference engine
        feedback_result = await preference_engine.submit_feedback(
            user_id=user_id,
            recommendation_id=recommendation_id,
            feedback_type=feedback_type,
            feedback_details=feedback_details
        )
        
        # Update confidence thresholds based on feedback
        confidence_update = await confidence_manager.process_feedback(
            user_id=user_id,
            recommendation_id=recommendation_id,
            feedback_type=feedback_type,
            feedback_details=feedback_details
        )
        
        return {
            "message": "Feedback submitted successfully",
            "user_id": user_id,
            "recommendation_id": recommendation_id,
            "feedback_type": feedback_type,
            "learning_impact": feedback_result.get("learning_impact", "low"),
            "confidence_adjustments": confidence_update,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Feedback submission failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics", response_model=PersonalizationMetricsResponse)
async def get_personalization_metrics(
    preference_engine: UserPreferenceEngine = Depends(get_preference_engine),
    confidence_manager: AdaptiveConfidenceManager = Depends(get_confidence_manager)
):
    """
    Get comprehensive personalization system metrics.
    
    Returns detailed metrics from all personalization subsystems including
    preference learning, confidence management, and recommendation quality.
    """
    try:
        # Gather metrics from all components concurrently
        metrics_tasks = [
            preference_engine.get_performance_metrics(),
            confidence_manager.get_performance_metrics()
        ]
        
        preference_metrics, confidence_metrics = await asyncio.gather(
            *metrics_tasks, return_exceptions=True
        )
        
        # Handle any exceptions in metrics gathering
        if isinstance(preference_metrics, Exception):
            preference_metrics = {"error": str(preference_metrics)}
        if isinstance(confidence_metrics, Exception):
            confidence_metrics = {"error": str(confidence_metrics)}
        
        # Calculate recommendation system metrics
        recommendation_metrics = {
            "total_recommendations_generated": preference_metrics.get("total_recommendations", 0),
            "avg_personalization_score": preference_metrics.get("avg_personalization_score", 0.0),
            "recommendation_diversity": preference_metrics.get("avg_diversity_score", 0.0),
            "user_satisfaction_rate": preference_metrics.get("user_satisfaction_rate", 0.0)
        }
        
        # Calculate overall personalization quality
        personalization_quality = {
            "overall_quality_score": (
                preference_metrics.get("learning_quality_score", 0.0) +
                confidence_metrics.get("adaptation_quality_score", 0.0) +
                recommendation_metrics.get("avg_personalization_score", 0.0)
            ) / 3,
            "user_engagement_score": preference_metrics.get("user_engagement_score", 0.0),
            "model_accuracy": preference_metrics.get("model_accuracy", 0.0),
            "adaptation_speed": confidence_metrics.get("adaptation_speed", 0.0)
        }
        
        return PersonalizationMetricsResponse(
            timestamp=datetime.utcnow(),
            preference_engine=preference_metrics,
            confidence_manager=confidence_metrics,
            recommendation_system=recommendation_metrics,
            personalization_quality=personalization_quality
        )
        
    except Exception as e:
        logger.error(f"Failed to get personalization metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/profile/{user_id}")
async def delete_user_profile(
    user_id: str,
    confirmation: bool = Query(..., description="Confirmation required for deletion"),
    preference_engine: UserPreferenceEngine = Depends(get_preference_engine),
    confidence_manager: AdaptiveConfidenceManager = Depends(get_confidence_manager)
):
    """
    Delete user personality profile and all associated data.
    
    Permanently removes user profile, preferences, learning data,
    and confidence thresholds. Requires explicit confirmation.
    """
    try:
        if not confirmation:
            raise HTTPException(
                status_code=400,
                detail="Profile deletion requires explicit confirmation"
            )
        
        # Delete user profile from preference engine
        preference_deletion = await preference_engine.delete_user_profile(user_id)
        
        # Delete confidence thresholds
        confidence_deletion = await confidence_manager.delete_user_data(user_id)
        
        return {
            "message": "User profile deleted successfully",
            "user_id": user_id,
            "deletion_summary": {
                "profile_data_deleted": preference_deletion.get("deleted_items", 0),
                "confidence_data_deleted": confidence_deletion.get("deleted_items", 0),
                "total_data_removed": preference_deletion.get("deleted_items", 0) + confidence_deletion.get("deleted_items", 0)
            },
            "deletion_timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Profile deletion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Background task functions
async def _background_preference_learning(
    preference_engine: UserPreferenceEngine,
    user_id: str,
    interaction_history: List[Dict[str, Any]]
):
    """Execute preference learning in background."""
    try:
        await preference_engine.batch_learn_preferences(
            user_id=user_id,
            interaction_history=interaction_history
        )
        logger.info(f"Background preference learning completed for user {user_id}")
    except Exception as e:
        logger.error(f"Background preference learning failed for user {user_id}: {e}")


async def _background_model_optimization(
    preference_engine: UserPreferenceEngine,
    user_id: str
):
    """Execute model optimization in background."""
    try:
        await preference_engine.optimize_user_models(user_id=user_id)
        logger.info(f"Background model optimization completed for user {user_id}")
    except Exception as e:
        logger.error(f"Background model optimization failed for user {user_id}: {e}")


# Export router
__all__ = ["router"]