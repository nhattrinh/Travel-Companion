"""Simple metrics collector stub (Phase 2)."""
import time
from contextlib import contextmanager
from collections import defaultdict

_timers = defaultdict(list)

@contextmanager
def record_latency(name: str):
    start = time.perf_counter()
    try:
        yield
    finally:
        _timers[name].append((time.perf_counter() - start) * 1000.0)

def get_metrics_snapshot():  # pragma: no cover
    return {k: {
        "count": len(v),
        "avg_ms": (sum(v) / len(v)) if v else 0.0,
        "p95_ms": sorted(v)[int(len(v)*0.95)-1] if v else 0.0
    } for k, v in _timers.items()}
