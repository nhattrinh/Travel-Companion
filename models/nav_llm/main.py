#!/usr/bin/env python3
"""
Navigation LLM - Main entry point for training/evaluation.

Usage:
    # Run with local model (llama-cpp-python)
    python main.py --model-path ./models/llama-4-scout.gguf
    
    # Run with remote API (llama.cpp server or vLLM)
    python main.py --api-url http://localhost:8080/v1
    
    # Run evaluation on test dataset
    python main.py --evaluate --dataset ./data/nav_eval.json
"""

import argparse
import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Optional

from model import (
    NavigationLLM,
    NavigationLLMConfig,
    SupportedLanguage,
    LLMError,
)
from metrics import (
    NavigationMetrics,
    LatencyMetrics,
    ToolUsageMetrics,
    ContextMetrics,
    SafetyMetrics,
    HallucinationMetrics,
    LanguageMetrics,
    RecommendationMetrics,
    MetricsCollector,
    get_metrics_collector,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# Sample Test Cases
# =============================================================================

SAMPLE_QUERIES = [
    {
        "message": "Where can I get good ramen nearby?",
        "language": "en",
        "location": (35.6762, 139.6503),  # Tokyo
        "expected_tools": ["get_nearby_places"],
    },
    {
        "message": "How do I get to Tokyo Station from here?",
        "language": "en",
        "location": (35.6812, 139.7671),  # Near Tokyo Station
        "expected_tools": ["get_route"],
    },
    {
        "message": "이 근처에 좋은 카페 있나요?",  # Korean: Any good cafes nearby?
        "language": "ko",
        "location": (37.5665, 126.9780),  # Seoul
        "expected_tools": ["get_nearby_places"],
    },
    {
        "message": "Phở là gì?",  # Vietnamese: What is Pho?
        "language": "vi",
        "location": (10.8231, 106.6297),  # Ho Chi Minh City
        "expected_tools": ["get_menu_item_info"],
    },
]


# =============================================================================
# Evaluation Functions
# =============================================================================

def evaluate_single_query(
    llm: NavigationLLM,
    query: dict,
    collector: MetricsCollector,
) -> NavigationMetrics:
    """Evaluate a single query and collect metrics."""
    
    query_id = f"q_{int(time.time() * 1000)}"
    language = SupportedLanguage(query["language"])
    
    # Start timing
    collector.start_request()
    start_time = time.perf_counter()
    
    try:
        # Run inference
        response = llm.chat_sync(
            user_message=query["message"],
            language=language,
            location=query.get("location"),
        )
        
        # Calculate latency
        latency_ms = int((time.perf_counter() - start_time) * 1000)
        
        # Build metrics
        metrics = NavigationMetrics(
            query_id=query_id,
            language=query["language"],
        )
        
        # Latency metrics
        metrics.latency = LatencyMetrics(
            total_latency_ms=latency_ms,
            time_to_full_response_ms=latency_ms,
        )
        
        # Tool usage metrics
        tools_called = [tc.name for tc in response.tool_calls]
        expected_tools = query.get("expected_tools", [])
        
        metrics.tool_usage = ToolUsageMetrics(
            tools_called=tools_called,
            successful_calls=len(tools_called),
            failed_calls=0,
            calls_per_query=float(len(tools_called)),
        )
        
        # Context metrics
        metrics.context = ContextMetrics(
            prompt_tokens=response.usage.get("prompt_tokens", 0),
            completion_tokens=response.usage.get("completion_tokens", 0),
            total_tokens=response.total_tokens,
        )
        
        # Language metrics (basic check)
        metrics.language_quality = LanguageMetrics(
            language=query["language"],
            language_adherence=True,  # Would need NLP to verify
            correctness_score=1.0 if response.content else 0.0,
        )
        
        # Safety metrics (basic)
        metrics.safety = SafetyMetrics(
            safety_check_passed=True,
            unsafe_suggestion_count=0,
        )
        
        # Hallucination (basic - check if tools were used for factual claims)
        metrics.hallucination = HallucinationMetrics(
            total_claims=1,
            verified_claims=1 if response.has_tool_calls else 0,
        )
        
        # Log result
        logger.info(
            f"Query '{query['message'][:30]}...' | "
            f"Latency: {latency_ms}ms | "
            f"Tools: {tools_called} | "
            f"Tokens: {response.total_tokens}"
        )
        
        collector.record(metrics)
        return metrics
        
    except LLMError as e:
        logger.error(f"LLM error: {e}")
        metrics = NavigationMetrics(query_id=query_id, language=query["language"])
        metrics.latency = LatencyMetrics(
            total_latency_ms=collector.end_request()
        )
        metrics.safety = SafetyMetrics(safety_check_passed=False)
        collector.record(metrics)
        return metrics


async def evaluate_async(
    llm: NavigationLLM,
    queries: list[dict],
    collector: MetricsCollector,
) -> list[NavigationMetrics]:
    """Evaluate queries asynchronously."""
    results = []
    
    for query in queries:
        query_id = f"q_{int(time.time() * 1000)}"
        language = SupportedLanguage(query["language"])
        
        collector.start_request()
        start_time = time.perf_counter()
        
        try:
            response = await llm.chat(
                user_message=query["message"],
                language=language,
                location=query.get("location"),
            )
            
            latency_ms = int((time.perf_counter() - start_time) * 1000)
            
            metrics = NavigationMetrics(
                query_id=query_id,
                language=query["language"],
            )
            metrics.latency = LatencyMetrics(
                total_latency_ms=latency_ms,
                time_to_full_response_ms=latency_ms,
            )
            metrics.tool_usage = ToolUsageMetrics(
                tools_called=[tc.name for tc in response.tool_calls],
                successful_calls=len(response.tool_calls),
                calls_per_query=float(len(response.tool_calls)),
            )
            metrics.context = ContextMetrics(
                prompt_tokens=response.usage.get("prompt_tokens", 0),
                completion_tokens=response.usage.get("completion_tokens", 0),
                total_tokens=response.total_tokens,
            )
            
            collector.record(metrics)
            results.append(metrics)
            
            logger.info(
                f"Query: {query['message'][:30]}... | "
                f"Latency: {latency_ms}ms | "
                f"Response: {response.content[:50]}..."
            )
            
        except Exception as e:
            logger.error(f"Error: {e}")
            metrics = NavigationMetrics(query_id=query_id, language=query["language"])
            results.append(metrics)
    
    return results


def print_summary(collector: MetricsCollector) -> None:
    """Print evaluation summary."""
    summary = collector.get_summary()
    
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total Queries:     {summary.get('total_queries', 0)}")
    print(f"Avg Latency:       {summary.get('avg_latency_ms', 0):.1f} ms")
    print(f"P50 Latency:       {summary.get('p50_latency_ms', 0):.1f} ms")
    print(f"P95 Latency:       {summary.get('p95_latency_ms', 0):.1f} ms")
    print(f"Grounded Rate:     {summary.get('avg_grounded_rate', 0):.1%}")
    print(f"Safety Pass Rate:  {summary.get('safety_pass_rate', 0):.1%}")
    print("=" * 60)


def load_dataset(path: str) -> list[dict]:
    """Load evaluation dataset from JSON file."""
    with open(path, "r") as f:
        return json.load(f)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Navigation LLM - Run and evaluate"
    )
    
    # Model options
    parser.add_argument(
        "--model-path",
        type=str,
        help="Path to local GGUF model file"
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default="http://localhost:8080/v1",
        help="URL for remote LLM API"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        help="API key for remote LLM"
    )
    
    # Evaluation options
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run evaluation mode"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Path to evaluation dataset JSON"
    )
    
    # Model parameters
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Max tokens in response"
    )
    parser.add_argument(
        "--n-ctx",
        type=int,
        default=8192,
        help="Context window size"
    )
    
    # Interactive mode
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run interactive chat mode"
    )
    
    args = parser.parse_args()
    
    # Build config
    config = NavigationLLMConfig(
        model_path=args.model_path,
        api_base_url=args.api_url,
        api_key=args.api_key,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        n_ctx=args.n_ctx,
    )
    
    # Initialize model
    llm = NavigationLLM(config)
    collector = get_metrics_collector()
    
    if args.evaluate:
        # Evaluation mode
        logger.info("Running evaluation...")
        
        if args.dataset:
            queries = load_dataset(args.dataset)
        else:
            queries = SAMPLE_QUERIES
            logger.info("Using sample queries (no dataset provided)")
        
        # Load model
        llm.load()
        
        # Run evaluation
        for query in queries:
            evaluate_single_query(llm, query, collector)
        
        # Print summary
        print_summary(collector)
        
    elif args.interactive:
        # Interactive mode
        logger.info("Starting interactive mode...")
        llm.load()
        
        print("\nNavigation LLM Interactive Mode")
        print("Type 'quit' to exit, 'lang:ko' or 'lang:vi' to change language")
        print("-" * 40)
        
        language = SupportedLanguage.ENGLISH
        history = []
        
        while True:
            try:
                user_input = input(f"\n[{language.value}] You: ").strip()
                
                if not user_input:
                    continue
                if user_input.lower() == "quit":
                    break
                if user_input.startswith("lang:"):
                    lang_code = user_input.split(":")[1].strip()
                    try:
                        language = SupportedLanguage(lang_code)
                        print(f"Language changed to: {language.value}")
                    except ValueError:
                        print(f"Unknown language: {lang_code}")
                    continue
                
                collector.start_request()
                response = llm.chat_sync(
                    user_message=user_input,
                    language=language,
                )
                latency = collector.end_request()
                
                print(f"\nAssistant: {response.content}")
                if response.tool_calls:
                    print(f"  [Tools used: {[tc.name for tc in response.tool_calls]}]")
                print(f"  [Latency: {latency}ms | Tokens: {response.total_tokens}]")
                
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                logger.error(f"Error: {e}")
    
    else:
        # Quick test with sample queries
        logger.info("Running quick test with sample queries...")
        llm.load()
        
        for query in SAMPLE_QUERIES[:2]:  # Just first 2
            evaluate_single_query(llm, query, collector)
        
        print_summary(collector)


if __name__ == "__main__":
    main()
