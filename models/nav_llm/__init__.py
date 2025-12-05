"""
Travel Companion Navigation LLM Module.

Context & Location Aware Navigator based on Llama 4 with RAG integration.

This module provides:
- NavigationLLM: Core Llama 4-based model with tool calling
- Metrics for evaluation and monitoring
"""

from .model import (
    NavigationLLMConfig,
    SupportedLanguage,
    NavigationLLM,
    ToolCall,
    ToolResult,
    LLMResponse,
    ChatMessage,
    LLMError,
    NAVIGATION_TOOLS,
    SYSTEM_PROMPTS,
)

from .metrics import (
    MetricType,
    RouteQualityMetrics,
    RecommendationMetrics,
    ItineraryMetrics,
    LanguageMetrics,
    SafetyMetrics,
    HallucinationMetrics,
    LatencyMetrics,
    ToolUsageMetrics,
    ContextMetrics,
    NavigationMetrics,
    MetricsCollector,
    get_metrics_collector,
    calculate_path_deviation,
    calculate_category_diversity,
    haversine_distance,
)

from .tools import (
    get_nearby_places,
    get_route,
    geocode,
    reverse_geocode,
    get_menu_item_info,
    get_local_etiquette,
    execute_tool,
    execute_tool_sync,
    Place,
    Route,
    RouteStep,
    TOOL_REGISTRY,
)

__all__ = [
    # Model & Config
    "NavigationLLMConfig",
    "SupportedLanguage",
    "NavigationLLM",
    "NAVIGATION_TOOLS",
    "SYSTEM_PROMPTS",
    # Types
    "ToolCall",
    "ToolResult",
    "LLMResponse",
    "ChatMessage",
    "LLMError",
    # Tools
    "get_nearby_places",
    "get_route",
    "geocode",
    "reverse_geocode",
    "get_menu_item_info",
    "get_local_etiquette",
    "execute_tool",
    "execute_tool_sync",
    "Place",
    "Route",
    "RouteStep",
    "TOOL_REGISTRY",
    # Metrics
    "MetricType",
    "RouteQualityMetrics",
    "RecommendationMetrics",
    "ItineraryMetrics",
    "LanguageMetrics",
    "SafetyMetrics",
    "HallucinationMetrics",
    "LatencyMetrics",
    "ToolUsageMetrics",
    "ContextMetrics",
    "NavigationMetrics",
    "MetricsCollector",
    "get_metrics_collector",
    # Utility functions
    "calculate_path_deviation",
    "calculate_category_diversity",
    "haversine_distance",
]

__version__ = "1.0.0"

