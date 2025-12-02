# Research Notes: Multi-Modal AI Travel Companion

Purpose: Central placeholder for design explorations, provider evaluations (OCR, translation APIs), latency benchmarking methodology, and future enhancements.

## Initial Questions
- OCR provider trade-offs (accuracy vs latency) for EN↔JA/ES/VI.
- Translation API pricing and rate limits.
- Device-side preprocessing techniques for low light and motion blur.
- Geolocation accuracy impacts on contextual navigation hints.

## Benchmark Targets (from spec success criteria)
- Live translation overlay p95 ≤1.0s, p99 ≤1.5s.
- Phrase suggestion response p95 ≤300ms.
- POI assistance initial load p95 ≤800ms.

## Next Steps
- Populate with comparative metrics once prototype services are integrated.
- Add privacy & retention compliance notes after implementing purge logic.
