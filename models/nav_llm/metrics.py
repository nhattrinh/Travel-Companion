"""
Metrics for Navigation LLM evaluation and monitoring.

Implements metrics from the spec:
- Route quality (path similarity, travel time difference)
- Recommendation quality (relevance, diversity)
- Itinerary quality (constraint satisfaction)
- Language quality (correctness, fluency)
- Safety & hallucination control
- System metrics (latency, tool usage, context size)
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


class MetricType(str, Enum):
    """Types of metrics collected."""
    ROUTE_QUALITY = "route_quality"
    RECOMMENDATION_QUALITY = "recommendation_quality"
    ITINERARY_QUALITY = "itinerary_quality"
    LANGUAGE_QUALITY = "language_quality"
    SAFETY = "safety"
    HALLUCINATION = "hallucination"
    LATENCY = "latency"
    TOOL_USAGE = "tool_usage"
    CONTEXT_SIZE = "context_size"


# =============================================================================
# Route Quality Metrics
# =============================================================================

@dataclass
class RouteQualityMetrics:
    """Metrics for evaluating route/direction quality."""
    
    # Path similarity (avg deviation in meters from reference)
    path_deviation_m: float = 0.0
    
    # Relative travel time difference (% vs shortest route)
    travel_time_diff_pct: float = 0.0
    
    # Number of correct waypoints
    correct_waypoints: int = 0
    total_waypoints: int = 0
    
    # Step instruction quality
    instruction_clarity_score: float = 0.0  # 0-1
    
    @property
    def waypoint_accuracy(self) -> float:
        """Percentage of correct waypoints."""
        if self.total_waypoints == 0:
            return 0.0
        return self.correct_waypoints / self.total_waypoints
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "path_deviation_m": self.path_deviation_m,
            "travel_time_diff_pct": self.travel_time_diff_pct,
            "waypoint_accuracy": self.waypoint_accuracy,
            "instruction_clarity_score": self.instruction_clarity_score,
        }


def calculate_path_deviation(
    generated_points: list[tuple[float, float]],
    reference_points: list[tuple[float, float]]
) -> float:
    """
    Calculate average deviation between generated and reference paths.
    
    Args:
        generated_points: List of (lat, lon) from LLM response
        reference_points: List of (lat, lon) from reference route
        
    Returns:
        Average deviation in meters
    """
    if not generated_points or not reference_points:
        return float("inf")
    
    total_deviation = 0.0
    for gen_point in generated_points:
        # Find closest reference point
        min_dist = float("inf")
        for ref_point in reference_points:
            dist = haversine_distance(
                gen_point[0], gen_point[1],
                ref_point[0], ref_point[1]
            )
            min_dist = min(min_dist, dist)
        total_deviation += min_dist
    
    return total_deviation / len(generated_points)


def haversine_distance(
    lat1: float, lon1: float,
    lat2: float, lon2: float
) -> float:
    """Calculate distance between two points in meters using Haversine formula."""
    R = 6371000  # Earth radius in meters
    
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    
    a = (math.sin(delta_phi / 2) ** 2 +
         math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    return R * c


# =============================================================================
# Recommendation Quality Metrics
# =============================================================================

@dataclass
class RecommendationMetrics:
    """Metrics for POI recommendation quality."""
    
    # Relevance scores (0-1)
    relevance_scores: list[float] = field(default_factory=list)
    
    # Category diversity (unique categories / total recommendations)
    category_diversity: float = 0.0
    
    # User interaction metrics (for online evaluation)
    click_through_rate: float = 0.0
    save_to_favorites_rate: float = 0.0
    
    # Distance appropriateness
    avg_distance_m: float = 0.0
    within_radius_pct: float = 0.0
    
    @property
    def avg_relevance(self) -> float:
        if not self.relevance_scores:
            return 0.0
        return sum(self.relevance_scores) / len(self.relevance_scores)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "avg_relevance": self.avg_relevance,
            "category_diversity": self.category_diversity,
            "click_through_rate": self.click_through_rate,
            "save_to_favorites_rate": self.save_to_favorites_rate,
            "avg_distance_m": self.avg_distance_m,
            "within_radius_pct": self.within_radius_pct,
        }


def calculate_category_diversity(categories: list[str]) -> float:
    """Calculate category diversity score."""
    if not categories:
        return 0.0
    unique = len(set(categories))
    return unique / len(categories)


# =============================================================================
# Itinerary Quality Metrics
# =============================================================================

@dataclass
class ItineraryMetrics:
    """Metrics for multi-stop itinerary planning quality."""
    
    # Constraint satisfaction
    budget_satisfied: bool = True
    time_windows_satisfied: bool = True
    preferences_satisfied: bool = True
    
    # Efficiency
    total_travel_time_minutes: int = 0
    total_distance_m: float = 0.0
    
    # Completeness
    requested_stops: int = 0
    planned_stops: int = 0
    
    @property
    def constraint_satisfaction_rate(self) -> float:
        """Percentage of constraints satisfied."""
        constraints = [
            self.budget_satisfied,
            self.time_windows_satisfied,
            self.preferences_satisfied,
        ]
        return sum(constraints) / len(constraints)
    
    @property
    def stop_completion_rate(self) -> float:
        if self.requested_stops == 0:
            return 1.0
        return self.planned_stops / self.requested_stops
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "constraint_satisfaction_rate": self.constraint_satisfaction_rate,
            "stop_completion_rate": self.stop_completion_rate,
            "total_travel_time_minutes": self.total_travel_time_minutes,
            "total_distance_m": self.total_distance_m,
        }


# =============================================================================
# Language Quality Metrics
# =============================================================================

@dataclass
class LanguageMetrics:
    """Metrics for language quality (EN/KO/VI)."""
    
    language: str = "en"
    
    # Correctness score (0-1)
    correctness_score: float = 0.0
    
    # Fluency score (0-1)
    fluency_score: float = 0.0
    
    # Language adherence (responded in correct language)
    language_adherence: bool = True
    
    # Grammar errors detected
    grammar_errors: int = 0
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "language": self.language,
            "correctness_score": self.correctness_score,
            "fluency_score": self.fluency_score,
            "language_adherence": self.language_adherence,
            "grammar_errors": self.grammar_errors,
        }


# =============================================================================
# Safety & Hallucination Metrics
# =============================================================================

@dataclass
class SafetyMetrics:
    """Metrics for safety and content policy compliance."""
    
    # Unsafe suggestions (closed places, prohibited areas)
    unsafe_suggestion_count: int = 0
    
    # Policy violations detected
    policy_violations: list[str] = field(default_factory=list)
    
    # Safety check passed
    safety_check_passed: bool = True
    
    # Llama Guard score (if used)
    guard_score: Optional[float] = None
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "unsafe_suggestion_count": self.unsafe_suggestion_count,
            "policy_violations": self.policy_violations,
            "safety_check_passed": self.safety_check_passed,
            "guard_score": self.guard_score,
        }


@dataclass
class HallucinationMetrics:
    """Metrics for hallucination detection."""
    
    # Claims made in response
    total_claims: int = 0
    
    # Claims verified against retrieved context
    verified_claims: int = 0
    
    # Unverifiable claims
    unverifiable_claims: int = 0
    
    # False claims detected
    false_claims: int = 0
    
    @property
    def grounded_response_rate(self) -> float:
        """Proportion of claims backed by retrieved context."""
        if self.total_claims == 0:
            return 1.0
        return self.verified_claims / self.total_claims
    
    @property
    def hallucination_rate(self) -> float:
        """Proportion of false claims."""
        if self.total_claims == 0:
            return 0.0
        return self.false_claims / self.total_claims
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "total_claims": self.total_claims,
            "verified_claims": self.verified_claims,
            "grounded_response_rate": self.grounded_response_rate,
            "hallucination_rate": self.hallucination_rate,
        }


# =============================================================================
# System Metrics
# =============================================================================

@dataclass
class LatencyMetrics:
    """Response time metrics."""
    
    # Time to first token (ms)
    time_to_first_token_ms: int = 0
    
    # Time to full response (ms)
    time_to_full_response_ms: int = 0
    
    # Tool execution times
    tool_latencies_ms: dict[str, int] = field(default_factory=dict)
    
    # Total request latency
    total_latency_ms: int = 0
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "time_to_first_token_ms": self.time_to_first_token_ms,
            "time_to_full_response_ms": self.time_to_full_response_ms,
            "tool_latencies_ms": self.tool_latencies_ms,
            "total_latency_ms": self.total_latency_ms,
        }


@dataclass
class ToolUsageMetrics:
    """Metrics for tool/function usage."""
    
    # Tools called
    tools_called: list[str] = field(default_factory=list)
    
    # Tool call success rate
    successful_calls: int = 0
    failed_calls: int = 0
    
    # Average calls per query
    calls_per_query: float = 0.0
    
    @property
    def success_rate(self) -> float:
        total = self.successful_calls + self.failed_calls
        if total == 0:
            return 1.0
        return self.successful_calls / total
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "tools_called": self.tools_called,
            "success_rate": self.success_rate,
            "calls_per_query": self.calls_per_query,
        }


@dataclass
class ContextMetrics:
    """Metrics for context window usage."""
    
    # Token counts
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    
    # Context composition
    system_prompt_tokens: int = 0
    conversation_history_tokens: int = 0
    retrieved_context_tokens: int = 0
    
    # Context window utilization
    max_context_tokens: int = 8192
    
    @property
    def context_utilization(self) -> float:
        """Percentage of context window used."""
        return self.total_tokens / self.max_context_tokens
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "context_utilization": self.context_utilization,
        }


# =============================================================================
# Aggregated Metrics
# =============================================================================

@dataclass
class NavigationMetrics:
    """Aggregated metrics for a navigation query."""
    
    query_id: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    language: str = "en"
    
    # Component metrics
    route_quality: Optional[RouteQualityMetrics] = None
    recommendation: Optional[RecommendationMetrics] = None
    itinerary: Optional[ItineraryMetrics] = None
    language_quality: Optional[LanguageMetrics] = None
    safety: Optional[SafetyMetrics] = None
    hallucination: Optional[HallucinationMetrics] = None
    latency: Optional[LatencyMetrics] = None
    tool_usage: Optional[ToolUsageMetrics] = None
    context: Optional[ContextMetrics] = None
    
    def to_dict(self) -> dict[str, Any]:
        result = {
            "query_id": self.query_id,
            "timestamp": self.timestamp.isoformat(),
            "language": self.language,
        }
        
        if self.route_quality:
            result["route_quality"] = self.route_quality.to_dict()
        if self.recommendation:
            result["recommendation"] = self.recommendation.to_dict()
        if self.itinerary:
            result["itinerary"] = self.itinerary.to_dict()
        if self.language_quality:
            result["language_quality"] = self.language_quality.to_dict()
        if self.safety:
            result["safety"] = self.safety.to_dict()
        if self.hallucination:
            result["hallucination"] = self.hallucination.to_dict()
        if self.latency:
            result["latency"] = self.latency.to_dict()
        if self.tool_usage:
            result["tool_usage"] = self.tool_usage.to_dict()
        if self.context:
            result["context"] = self.context.to_dict()
        
        return result


# =============================================================================
# Metrics Collector
# =============================================================================

class MetricsCollector:
    """Collects and aggregates navigation metrics."""
    
    def __init__(self):
        self._metrics: list[NavigationMetrics] = []
        self._start_time: Optional[float] = None
    
    def start_request(self) -> None:
        """Mark the start of a request for latency tracking."""
        self._start_time = time.perf_counter()
    
    def end_request(self) -> int:
        """Mark the end of a request, return total latency in ms."""
        if self._start_time is None:
            return 0
        elapsed = time.perf_counter() - self._start_time
        self._start_time = None
        return int(elapsed * 1000)
    
    def record(self, metrics: NavigationMetrics) -> None:
        """Record a metrics snapshot."""
        self._metrics.append(metrics)
        logger.info(f"Recorded metrics for query {metrics.query_id}")
    
    def get_summary(self, last_n: int = 100) -> dict[str, Any]:
        """Get summary statistics for recent queries."""
        recent = self._metrics[-last_n:]
        if not recent:
            return {}
        
        # Calculate averages
        latencies = [
            m.latency.total_latency_ms
            for m in recent
            if m.latency
        ]
        grounded_rates = [
            m.hallucination.grounded_response_rate
            for m in recent
            if m.hallucination
        ]
        safety_passed = [
            m.safety.safety_check_passed
            for m in recent
            if m.safety
        ]
        
        return {
            "total_queries": len(recent),
            "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0,
            "p50_latency_ms": sorted(latencies)[len(latencies) // 2] if latencies else 0,
            "p95_latency_ms": sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0,
            "avg_grounded_rate": sum(grounded_rates) / len(grounded_rates) if grounded_rates else 0,
            "safety_pass_rate": sum(safety_passed) / len(safety_passed) if safety_passed else 0,
        }
    
    def clear(self) -> None:
        """Clear all recorded metrics."""
        self._metrics.clear()


# Global metrics collector instance
_collector = MetricsCollector()


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector."""
    return _collector
