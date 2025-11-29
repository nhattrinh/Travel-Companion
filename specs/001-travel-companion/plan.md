# Implementation Plan: Multi-Modal AI Travel Companion

**Branch**: `001-travel-companion` | **Date**: 2025-11-14 | **Spec**: `specs/001-travel-companion/spec.md`
**Input**: Feature specification (multi-modal travel assistance across translation, navigation, phrasebook, trip memory)

## Summary
Deliver a FastAPI backend + native iOS (SwiftUI) client providing live camera translation overlay, context-aware navigation guidance, smart phrase suggestions, and persistent trip memory. Prioritize anxiety reduction via rapid comprehension and just-in-time context.

## Technical Context
**Language/Version**: Python 3.11 (backend), Swift 5.9 (iOS 17+)
**Primary Dependencies**: FastAPI, SQLAlchemy + Alembic, Redis (optional caching), Map provider SDK/API (Mapbox or OSM wrapper), OCR & Translation external APIs, SwiftUI, CoreLocation, AVFoundation/VisionKit
**Storage**: PostgreSQL (primary), Redis (caching), iOS Core Data (local cache only)
**Testing**: pytest (unit/integration/perf smoke), Xcode tests (unit, snapshot/UI)
**Target Platform**: Backend: Linux container; iOS: iOS 17+ devices/simulator
**Project Type**: Multi-service (backend + native client)
**Performance Goals**: Live overlay p95 ≤1.0s (p99 ≤1.5s); Phrase suggestion p95 ≤300ms; Navigation initial load p95 ≤800ms; Favorites retrieval p95 ≤200ms
**Constraints**: Limited offline (favorites & phrases only); Retention ≤30 days for sensitive frames/location; Envelope response format; >80% coverage for new backend modules
**Scale/Scope**: Initial city-level POI usage; thousands of phrases; moderate concurrent users (pilot); extensible language pairs

## Constitution Check
1. Code Quality Discipline: Test plan defined (unit/integration/perf smoke), coverage target set. Error handling -> envelope.
2. User Experience Consistency: Response envelope `{status,data,error}` mandated; deprecation path tasks added (T151–T152).
3. Performance & Efficiency Standards: Latency budgets defined; instrumentation tasks (T149–T150) planned.
4. Security & Reliability Constraints: Retention & purge (T144–T146, T108–T109); secret management via `.env`.
5. Workflow & Quality Gates: CI pipeline draft (T018); gating tasks Phase 8 (T121–T139).

No waivers requested.

## Project Structure

### Documentation
```
specs/001-travel-companion/
├── spec.md
├── plan.md
├── tasks.md
├── research.md (T001 placeholder)
├── data-model.md (placeholder)
└── contracts/ (future API contract expansions)
```

### Backend
```
app/
├── api/
│   ├── auth_endpoints.py
│   ├── translation_endpoints.py
│   ├── navigation_endpoints.py
│   ├── phrasebook_endpoints.py
│   ├── trips_endpoints.py
│   ├── privacy_endpoints.py (T144)
│   └── health_endpoints.py
├── core/
│   ├── db.py
│   ├── security.py / jwt.py
│   ├── logging.py
│   ├── metrics.py (+ metrics_translation.py, metrics_navigation_phrase.py)
│   ├── cache_client.py
│   ├── error_handlers.py
│   ├── deprecation.py (T151)
│   └── validation.py
├── config/
│   ├── settings.py
│   └── loader.py
├── models/
│   ├── user.py
│   ├── translation.py
│   ├── poi.py
│   ├── phrase.py
│   ├── favorite.py
│   ├── trip.py
│   ├── phrase_suggestion.py (T148)
│   └── __init__.py
├── schemas/
│   ├── base.py
│   ├── user.py
│   ├── translation.py
│   ├── poi.py
│   ├── phrase.py
│   ├── favorite.py
│   ├── trip.py
│   └── __init__.py
├── services/
│   ├── ai_ocr_service.py
│   ├── ai_translation_service.py
│   ├── image_preprocess.py
│   ├── navigation_service.py
│   ├── maps_client.py
│   ├── phrase_suggestion_service.py
│   ├── translation_history_service.py
│   ├── trip_service.py
│   ├── privacy_purge_service.py (T145)
│   └── etiquette_data.py
└── middleware/
    ├── request_context.py
    ├── rate_limit.py
    └── auth.py (existing/extended)
```

### iOS
```
ios/TravelCompanion/
├── TravelCompanionApp.swift
├── Shared/
│   ├── Config/Environment.swift
│   ├── Models/(UserDTO, TranslationDTO, POIDTO, PhraseDTO, FavoriteDTO, TripDTO)
│   ├── Utilities/(ImageEncoding.swift, LocalizationHelper.swift)
│   └── Persistence/(CoreData stack, entities)
├── Security/KeychainTokenStore.swift
├── Services/(APIClient.swift, AuthService.swift, TranslationService.swift, NavigationService.swift, PhrasebookService.swift, TripService.swift)
├── Features/
│   ├── Auth/(LoginView.swift, RegisterView.swift)
│   ├── CameraTranslate/(CameraView.swift, OverlayRenderer.swift, StaticCaptureView.swift)
│   ├── Navigation/(MapView.swift, POIDetailView.swift, NavigationViewModel.swift)
│   ├── Phrasebook/(PhraseListView.swift, ChatSuggestionView.swift, PhrasebookViewModel.swift)
│   └── TripOverview/(TripOverviewView.swift, RecentTranslationsView.swift, TripOverviewViewModel.swift)
└── Accessibility/Audit.md
```

## Complexity Tracking
_No constitution violations; architecture intentionally minimal (single backend service + iOS client)._ 

## Risk & Mitigation
| Risk | Mitigation |
|------|------------|
| External OCR latency | Caching segments, async batching, profiling (T149) |
| Map provider rate limits | POI caching (T071) + feature toggles |
| Data retention sensitivity | Purge job (T108/T109), audit tests (future) |
| Latency budget overruns | Segment timers & perf smoke tests (T060, T131) |

## Initial Data Model (Summary)
Tables: users, translations, phrases, favorites, trips, pois, phrase_suggestions (+ indexes: user/time, geo coordinates). Detailed schema to be expanded in `data-model.md`.

## Initial API Surface (Draft)
Auth: POST /auth/register, /auth/login, /auth/refresh
Translation: POST /translation/live-frame, POST /translation/image, POST /translation/text, GET /translation/history
Phrasebook: GET /phrases, POST /phrases, POST /phrases/{id}/favorite, GET /phrases/favorites, GET /phrases/suggestions
Navigation: GET /navigation/pois, GET /navigation/context
Trips: POST /trips/start, POST /trips/{id}/end, GET /trips/{id}/summary
Privacy: POST /privacy/purge
Health/Metrics: GET /health, GET /metrics

## Performance Budget Breakdown (US1)
Upload (≤300ms), OCR (≤600ms), Translation (≤600ms, overlapped), Assembly (≤100ms) → optimize concurrency to meet ≤1000ms p95.

## Incremental Delivery
1. Phases 1–2 (Setup + Foundational) → Auth baseline
2. Phase 3 (US1) → MVP translation overlay
3. Phase 4 (US2) → Navigation & context hints
4. Phase 5 (US3) → Phrasebook & suggestions
5. Phase 6 (US4) → Trip memory & purge
6. Phase 7 (Polish), Phase 8 (Gates)

## Observability Plan
Structured logging (JSON), correlation id middleware, metrics timers (translation, POI, phrase suggestion), latency & error rate dashboards (future). 

## Security Notes
JWT auth, input validation (size/type for images), secret management via env, encryption at rest (PostgreSQL + object storage) tasks scheduled, purge job for retention.

## Pending Clarifications
(None – remediation tasks added to address gaps.)

## Next Steps
Proceed with prerequisite check, then begin Phase 1 tasks (T001–T020).
