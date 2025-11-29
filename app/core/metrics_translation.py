"""Translation latency instrumentation utilities.

Task: T149
Provides simple timing context manager and aggregation hooks.
"""
import time
from contextlib import contextmanager

_translation_timings_ms: list[float] = []


@contextmanager
def record_translation_latency():
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        _translation_timings_ms.append(elapsed_ms)


def snapshot_latency_stats() -> dict:
    if not _translation_timings_ms:
        return {"count": 0, "p95_ms": None, "p99_ms": None}
    sorted_vals = sorted(_translation_timings_ms)
    count = len(sorted_vals)

    def _percentile(p: float) -> float:
        idx = int(round(p * (count - 1)))
        return sorted_vals[idx]
    return {
        "count": count,
        "p95_ms": _percentile(0.95),
        "p99_ms": _percentile(0.99),
    }
