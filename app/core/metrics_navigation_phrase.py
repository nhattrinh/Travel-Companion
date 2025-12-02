"""Navigation & phrase suggestion metrics.

Task: T150
Collects latency and cache hit stats for POI & phrase suggestions.
"""
import time
from contextlib import contextmanager

_poi_timings_ms: list[float] = []
_phrase_timings_ms: list[float] = []
_phrase_cache_hits: int = 0
_phrase_cache_misses: int = 0


@contextmanager
def record_poi_latency():
    start = time.perf_counter()
    try:
        yield
    finally:
        _poi_timings_ms.append((time.perf_counter() - start) * 1000.0)


@contextmanager
def record_phrase_latency(cache_hit: bool = False):
    global _phrase_cache_hits, _phrase_cache_misses
    start = time.perf_counter()
    try:
        yield
    finally:
        _phrase_timings_ms.append((time.perf_counter() - start) * 1000.0)
        if cache_hit:
            _phrase_cache_hits += 1
        else:
            _phrase_cache_misses += 1


def _percentiles(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "p95_ms": None, "p99_ms": None}
    vals = sorted(values)
    count = len(vals)

    def _p(p: float) -> float:
        idx = int(round(p * (count - 1)))
        return vals[idx]
    return {"count": count, "p95_ms": _p(0.95), "p99_ms": _p(0.99)}


def snapshot_metrics() -> dict:
    return {
        "poi": _percentiles(_poi_timings_ms),
        "phrase": _percentiles(_phrase_timings_ms),
        "phrase_cache": {
            "hits": _phrase_cache_hits,
            "misses": _phrase_cache_misses,
        },
    }
