# Tasks: Multi-Modal AI Travel Companion

**Input**: Design documents from `/specs/001-travel-companion/`
**Prerequisites**: `spec.md` (required), future `plan.md` (architecture), research, data-model, contracts (to be generated)

**Tests**: Test tasks included (unit, integration, performance smoke). Write tests first where indicated.
**Organization**: Tasks grouped by user story (US1–US4) after Setup & Foundational phases for independent delivery.

## Format: `[ID] [P?] [Story] Description`
- `[P]`: Parallelizable (different files, no unmet dependencies)
- `[US#]`: User story label (only in story phases)
- Every task includes a concrete file path or artifact.

---
## Phase 1: Setup (Shared Infrastructure)
**Purpose**: Repository + base project scaffolding for backend (FastAPI) and iOS (SwiftUI), environment & tooling.

- [X] T001 Initialize feature documentation directory structure in `specs/001-travel-companion/` (research.md, data-model.md placeholders)
- [X] T002 Create backend service skeleton in `app/api/auth_endpoints.py`, `app/api/__init__.py`
- [X] T003 [P] Add new backend modules directories: `app/schemas/`, `app/services/`, `app/models/` (init files)
- [X] T004 [P] Add iOS project root `ios/TravelCompanion/` with `TravelCompanionApp.swift`
- [X] T005 Create `ios/TravelCompanion/Shared/Config/Environment.swift` for base URLs
- [X] T006 [P] Add `.env.example` with required vars (POSTGRES_URL, REDIS_URL, JWT_SECRET, API_KEYS) at repository root
- [X] T007 Add `docker-compose.travel.yml` including FastAPI service, Postgres, Redis
- [X] T008 [P] Extend `Dockerfile` to include FastAPI dependencies (OCR/translation libs) and multi-stage build notes
- [X] T009 Add `Makefile` targets: `dev-backend`, `lint`, `migrate`, `test`, `perf-smoke`
- [X] T010 [P] Create `app/core/db.py` with SQLAlchemy engine + session factory
- [X] T011 [P] Create `app/core/security.py` with JWT issue/verify stubs
- [X] T012 Add `app/config/settings.py` updated for travel companion variables
- [X] T013 [P] Implement `app/middleware/request_context.py` for correlation id & timing
- [X] T014 Add `README-section-travel.md` describing architecture & setup (will merge later)
- [X] T015 [P] Set up iOS Xcode project configuration file `ios/TravelCompanion/Project.xcconfig`
- [X] T016 Add root `requirements-travel.txt` (merge/update existing `requirements.txt` later)
- [X] T017 [P] Create initial `alembic/` migration environment (env.py, script.py.mako)
- [X] T018 Add GitHub Actions pipeline draft `.github/workflows/travel-companion-ci.yml`
- [X] T019 [P] Create `ios/TravelCompanion/Services/APIClient.swift` (URLSession wrapper skeleton)
- [X] T020 Add `docs/architecture/travel-companion-overview.md` high-level system diagram placeholder

**Checkpoint**: Backend & iOS skeleton exist; env + build tooling ready.

---
## Phase 2: Foundational (Blocking Prerequisites)
**Purpose**: Core infrastructure required before any story implementation.
**CRITICAL**: Must complete before US1–US4.

- [ ] T021 Implement User model SQLAlchemy in `app/models/user.py`
- [ ] T022 [P] Implement BaseSchema DTOs in `app/schemas/base.py`
- [ ] T023 [P] Implement UserCreate, UserRead schemas in `app/schemas/user.py`
- [ ] T024 Add password hashing utility in `app/core/security.py` (bcrypt/argon2)
- [ ] T025 [P] Implement JWT generate/refresh logic `app/core/jwt.py`
- [ ] T026 Add auth endpoints (register/login/refresh) in `app/api/auth_endpoints.py`
- [ ] T027 [P] Create Alembic migration for users table `alembic/versions/xxxx_create_users.py`
- [ ] T028 Implement global error handlers in `app/core/error_handlers.py` for envelope responses
- [ ] T029 [P] Implement configuration loader + validation in `app/config/loader.py`
- [ ] T030 Implement health endpoint in `app/api/health_endpoints.py`
- [ ] T031 [P] Integrate logging configuration (structured) in `app/core/logging.py`
- [ ] T032 Set up Redis client in existing `app/core/cache_client.py` (extend if needed)
- [ ] T033 [P] Create metrics collector stub `app/core/metrics.py`
- [ ] T034 Add base iOS models `ios/TravelCompanion/Shared/Models/UserDTO.swift`
- [ ] T035 [P] Implement iOS Keychain token store in `ios/TravelCompanion/Security/KeychainTokenStore.swift`
- [ ] T036 Implement iOS AuthService `ios/TravelCompanion/Services/AuthService.swift`
- [ ] T037 Add iOS Onboarding/Login views `ios/TravelCompanion/Features/Auth/LoginView.swift`
- [ ] T038 [P] Backend unit tests for auth + security `tests/unit/test_auth_security.py`
- [ ] T039 Integration tests for auth endpoints `tests/integration/test_auth_endpoints.py`
- [ ] T040 [P] Add principle checklist doc placeholder `docs/principles/checklist-travel.md`

**Checkpoint**: Auth working end-to-end; metrics, logging, error handling baseline.

---
## Phase 3: User Story 1 – Live Camera Text Translation Overlay (Priority: P1) 🎯 MVP
**Goal**: Real-time menu/sign translation overlay.
**Independent Test**: Upload image frame → receive translated segments overlay JSON; can save translation.

### Tests (Write First)
- [ ] T041 [P] [US1] Unit test OCR preprocessor in `tests/unit/test_ocr_preprocess.py`
- [ ] T042 [P] [US1] Unit test translation service basic path `tests/unit/test_translation_service.py`
- [ ] T043 [US1] Integration test live-frame endpoint `tests/integration/test_translation_live_frame.py`

### Implementation
- [ ] T044 [P] [US1] Create Translation & OCR models in `app/models/translation.py`
- [ ] T045 [P] [US1] Add translation schemas in `app/schemas/translation.py`
- [ ] T046 [US1] Implement OCR service `app/services/ai_ocr_service.py` (stub external call)
- [ ] T047 [US1] Implement translation service `app/services/ai_translation_service.py`
- [ ] T048 [P] [US1] Implement image preprocessing utilities `app/services/image_preprocess.py`
- [ ] T049 [US1] Create live frame endpoint `app/api/translation_endpoints.py` (multipart handler)
- [ ] T050 [P] [US1] Implement language detection helper `app/services/lang_detect.py`
- [ ] T051 [US1] Add translation history persistence logic `app/services/translation_history_service.py`
- [ ] T052 [P] [US1] Alembic migration for translations table `alembic/versions/xxxx_create_translations.py`
- [ ] T053 [US1] iOS Camera module `ios/TravelCompanion/Features/CameraTranslate/CameraView.swift`
- [ ] T054 [P] [US1] iOS Overlay renderer `ios/TravelCompanion/Features/CameraTranslate/OverlayRenderer.swift`
- [ ] T055 [US1] iOS TranslationViewModel `ios/TravelCompanion/Features/CameraTranslate/TranslationViewModel.swift`
- [ ] T056 [P] [US1] iOS ImageEncoding utility `ios/TravelCompanion/Shared/Utilities/ImageEncoding.swift`
- [ ] T057 [US1] Add save translation action in `app/api/translation_endpoints.py`
- [ ] T058 [US1] Implement envelope-consistent error handling for OCR failures
- [ ] T059 [US1] Unit test envelope error patterns `tests/unit/test_error_envelope.py`
- [ ] T060 [US1] Performance smoke test translation path `tests/perf/test_translation_latency.py`

**Checkpoint**: US1 independently deployable & testable; overlay workflow operational.

---
## Phase 4: User Story 2 – Context-Aware Navigation Assistance (Priority: P2)
**Goal**: Nearby POIs + contextual etiquette & transit hints.
**Independent Test**: User queries POIs; receives list with distances & context notes.

### Tests (Write First)
- [ ] T061 [P] [US2] Unit test navigation service POI normalization `tests/unit/test_navigation_normalize.py`
- [ ] T062 [P] [US2] Integration test POI fetch endpoint `tests/integration/test_navigation_pois.py`
- [ ] T063 [US2] Unit test etiquette/context builder `tests/unit/test_context_builder.py`

### Implementation
- [ ] T064 [P] [US2] Add POI model `app/models/poi.py`
- [ ] T065 [P] [US2] Add POI schemas `app/schemas/poi.py`
- [ ] T066 [US2] Implement navigation service `app/services/navigation_service.py`
- [ ] T067 [US2] Add maps provider client `app/services/maps_client.py`
- [ ] T068 [P] [US2] Alembic migration for POIs table `alembic/versions/xxxx_create_pois.py`
- [ ] T069 [US2] Implement POI endpoint `app/api/navigation_endpoints.py`
- [ ] T070 [P] [US2] Implement context etiquette rules dataset `app/services/etiquette_data.py`
- [ ] T071 [US2] Caching layer for POI queries in `app/core/cache_client.py`
- [ ] T072 [US2] iOS NavigationViewModel `ios/TravelCompanion/Features/Navigation/NavigationViewModel.swift`
- [ ] T073 [P] [US2] iOS Map screen `ios/TravelCompanion/Features/Navigation/MapView.swift`
- [ ] T074 [US2] iOS POIDetailSheet `ios/TravelCompanion/Features/Navigation/POIDetailView.swift`
- [ ] T075 [US2] Handle offline state: disable POI fetch, display reconnect banner (no POI fallback per limited offline scope)
- [ ] T076 [US2] Integration test etiquette notes inclusion `tests/integration/test_navigation_context.py`
- [ ] T077 [P] [US2] Performance smoke test POI endpoint `tests/perf/test_poi_latency.py`

**Checkpoint**: US2 functional independently (POI listing & context hints) with caching.

---
## Phase 5: User Story 3 – Smart Contextual Phrasebook & Chat Suggestions (Priority: P3)
**Goal**: Context-driven phrase suggestions & favorites management.
**Independent Test**: User selects context → receives suggestions; saves phrases; no navigation required.

### Tests (Write First)
- [ ] T078 [P] [US3] Unit test phrase suggestion scoring `tests/unit/test_phrase_scoring.py`
- [ ] T079 [P] [US3] Integration test phrase suggestions endpoint `tests/integration/test_phrase_suggestions.py`
- [ ] T080 [US3] Unit test favorites toggle endpoint `tests/unit/test_favorites_toggle.py`

### Implementation
- [ ] T081 [P] [US3] Add Phrase model `app/models/phrase.py`
- [ ] T082 [P] [US3] Add Favorite model `app/models/favorite.py`
- [ ] T083 [US3] Phrase schemas `app/schemas/phrase.py`
- [ ] T084 [US3] Favorite schemas `app/schemas/favorite.py`
- [ ] T085 [P] [US3] Alembic migration phrases table `alembic/versions/xxxx_create_phrases.py`
- [ ] T086 [P] [US3] Alembic migration favorites table `alembic/versions/xxxx_create_favorites.py`
- [ ] T087 [US3] Implement phrase suggestion service `app/services/phrase_suggestion_service.py`
- [ ] T088 [US3] Implement phrasebook endpoints `app/api/phrasebook_endpoints.py`
- [ ] T089 [P] [US3] Implement favorites endpoints in same router
- [ ] T090 [US3] iOS PhrasebookViewModel `ios/TravelCompanion/Features/Phrasebook/PhrasebookViewModel.swift`
- [ ] T091 [P] [US3] iOS PhraseListView `ios/TravelCompanion/Features/Phrasebook/PhraseListView.swift`
- [ ] T092 [US3] iOS ChatSuggestionView `ios/TravelCompanion/Features/Phrasebook/ChatSuggestionView.swift`
- [ ] T093 [US3] Caching frequent phrases (Redis) in phrase suggestion service
- [ ] T094 [US3] Integration test favorites persistence `tests/integration/test_favorites_persistence.py`
- [ ] T095 [US3] Performance smoke test phrase suggestion latency `tests/perf/test_phrase_latency.py`

**Checkpoint**: US3 delivers context suggestions & favorites management independently.

---
## Phase 6: User Story 4 – Persistent Trip Memory (Favorites & History) (Priority: P4)
**Goal**: Trip lifecycle, history retrieval, favorites & places continuity.
**Independent Test**: Create trip → perform translations → retrieve history & favorites next session.

### Tests (Write First)
- [ ] T096 [P] [US4] Unit test trip lifecycle service `tests/unit/test_trip_service.py`
- [ ] T097 [P] [US4] Integration test trip summary endpoint `tests/integration/test_trip_summary.py`
- [ ] T098 [US4] Unit test translation history retrieval `tests/unit/test_translation_history.py`

### Implementation
- [ ] T099 [P] [US4] Add Trip model `app/models/trip.py`
- [ ] T100 [US4] Trip schemas `app/schemas/trip.py`
- [ ] T101 [P] [US4] Alembic migration trips table `alembic/versions/xxxx_create_trips.py`
- [ ] T102 [US4] Trip service `app/services/trip_service.py`
- [ ] T103 [US4] Trip endpoints `app/api/trips_endpoints.py`
- [ ] T104 [US4] Extend translation history service to filter by trip
- [ ] T105 [US4] iOS TripOverviewViewModel `ios/TravelCompanion/Features/TripOverview/TripOverviewViewModel.swift`
- [ ] T106 [P] [US4] iOS TripOverviewView `ios/TravelCompanion/Features/TripOverview/TripOverviewView.swift`
- [ ] T107 [US4] iOS RecentTranslationsView `ios/TravelCompanion/Features/TripOverview/RecentTranslationsView.swift`
- [ ] T108 [US4] Backend purge job scheduler (30-day retention) `app/services/purge_job.py`
- [ ] T109 [US4] Integration test retention purge logic `tests/integration/test_retention_purge.py`
- [ ] T110 [US4] Performance smoke test trip summary retrieval `tests/perf/test_trip_summary_latency.py`

**Checkpoint**: US4 completes persistent memory features.

---
## Phase 7: Cross-Cutting Polish & Optimization
**Purpose**: Improve quality, performance, security & UX consistency across stories.

- [ ] T111 [P] Documentation updates in `docs/architecture/travel-companion-overview.md` (add final diagrams)
- [ ] T112 Code cleanup & remove dead code across `app/services/` & iOS ViewModels
- [ ] T113 Performance optimization translation pipeline profiling report `docs/perf/translation-profile.md`
- [ ] T114 [P] Additional unit tests for edge cases `tests/unit/test_edge_cases.py`
- [ ] T115 Security hardening (input validation) `app/core/validation.py`
- [ ] T116 Run privacy purge endpoint tests `tests/integration/test_privacy_purge.py`
- [ ] T117 [P] Accessibility review iOS views `ios/TravelCompanion/Accessibility/Audit.md`
- [ ] T118 Add rate limiting middleware `app/middleware/rate_limit.py` tuning
- [ ] T119 [P] Implement metrics endpoint `app/api/metrics_endpoints.py`
- [ ] T120 Refine Redis cache invalidation strategy `docs/cache/invalidation-strategy.md`

---
## Phase 8: Principle Alignment Tasks (Release Gates)

### Code Quality Discipline
- [ ] T121 QUAL001 Verify >80% coverage in new backend modules (`coverage-report.txt`)
- [ ] T122 QUAL002 Lint & format pass (`make lint`) clean
- [ ] T123 QUAL003 Remove unused imports/dead code diff check
- [ ] T124 QUAL004 Docstrings for public services & routers audit

### User Experience Consistency
- [ ] T125 UX001 Confirm envelope `{status,data,error}` across all responses (script check)
- [ ] T126 UX002 Deprecated fields list (if any) `docs/deprecations.md`
- [ ] T127 UX003 Translation failure & low confidence messages reviewed
- [ ] T128 UX004 Phrase suggestions phrasing clarity audit

### Performance & Efficiency
- [ ] T129 PERF001 Capture baseline vs final p95/p99 metrics `docs/perf/baseline-vs-final.md`
- [ ] T130 PERF002 Profile heavy OCR & translation calls summary
- [ ] T131 PERF003 Verify p95 overlay latency ≤1.0s test report
- [ ] T132 PERF004 Review cache TTLs & invalidation configuration

### Security & Reliability
- [ ] T133 SEC001 Secret scan results (`scan-report.txt`) zero leaks
- [ ] T134 SEC002 Input validation coverage mapping
- [ ] T135 SEC003 Health checks cover DB, Redis, external APIs
- [ ] T136 SEC004 Dependency security patch audit list

### Workflow & Quality Gates
- [ ] T137 FLOW001 CI pipeline passes all stages
- [ ] T138 FLOW002 Reviewer checklist confirmations captured
- [ ] T139 FLOW003 Waivers documented & none expired `WAIVERS.md`

---
## Dependencies & Execution Order

### Phase Dependencies
- Setup (Phase 1): No dependencies.
- Foundational (Phase 2): Depends on Setup completion; blocks US1–US4.
- US1–US4 (Phases 3–6): Each depends on Foundational; can proceed in parallel after Phase 2.
- Polish (Phase 7): Depends on desired user stories completion (at least US1 for MVP, ideally all).
- Principle Alignment (Phase 8): Final gate after all functional phases.

### User Story Independence
- US1: Only needs auth & core infrastructure.
- US2: Independent except requires auth and baseline config; no dependence on US1 logic.
- US3: Independent; uses auth & can run without navigation.
- US4: Uses data produced by US1–US3 but can start after translations & phrase tables exist; still testable standalone with stub data.

### Within Story Ordering
1. Write tests → ensure failing state → implement services/models → endpoints → integration tests → performance smoke.
2. Parallel tasks marked [P] can proceed after their blocking predecessor models or schemas exist.

### Parallel Opportunities
- Infrastructure directory creations & initial skeletons (T002–T005, T010–T013).
- All Alembic migrations in distinct files (translations, POIs, phrases, favorites, trips) can be authored concurrently once models drafted.
- iOS view implementations per feature can progress parallel to backend endpoint development.
- Performance smoke tests can run parallel after endpoints stable.

---
## Parallel Example: User Story 1
```
# Parallel group 1 (models & schemas):
T044 Create Translation model
T045 Translation schemas
T048 Image preprocessing utilities
T050 Language detection helper

# Parallel group 2 (iOS components):
T053 CameraView
T054 OverlayRenderer
T056 ImageEncoding utility
```

---
## Implementation Strategy

### MVP First (US1 Only)
1. Complete Phase 1 & 2.
2. Execute Phase 3 tasks for US1.
3. Run tests & perf smoke (T041–T043, T060).
4. Deploy demo focusing on camera overlay translation.

### Incremental Delivery
1. Add US2 (navigation) parallel to finalizing US1 performance tweaks.
2. Add US3 (phrasebook & suggestions) with caching.
3. Add US4 (trip memory) and retention purge job.
4. Polish & Gate tasks ensure stability before public release.

### Performance & Quality Gate
- Collect baseline metrics after US1; re-collect after US2 & US3 integrations.
- Final auditing tasks (T121–T139) must pass before tagging release.

### Rollback & Risk Mitigation
- Feature toggles for navigation & phrase suggestions to disable if external API instability.
- Rate limiting reduces overload risk on translation endpoint.
- Purge job ensures retention compliance.

---
## Task Counts Summary
- Setup: 20
- Foundational: 20 (T021–T040)
- US1: 20 (T041–T060)
- US2: 17 (T061–T077)
- US3: 18 (T078–T095)
- US4: 12 (T096–T107)
- Additional US4 tasks (retention & tests): 3 (T108–T110)
- Polish: 10 (T111–T120)
- Principles Alignment: 19 (T121–T139)
Total: 139

**MVP Scope**: Complete through T060 (Phases 1–3) → deliver live translation overlay.

**All tasks follow checklist format requirements.**

---
## Remediation Additions (Consistency Gaps)

### User Story 1 / Core
- [ ] T140 [US1] Implement static image translation endpoint `app/api/translation_endpoints.py` (route: POST /translation/image)
- [ ] T141 [P] [US1] iOS StaticCaptureView for still image translation `ios/TravelCompanion/Features/CameraTranslate/StaticCaptureView.swift`

### User Profile & Preferences
- [ ] T142 Create user profile endpoint (language preferences) `app/api/user_profile_endpoints.py`
- [ ] T143 [P] Alembic migration add user preferences column JSONB to users `alembic/versions/xxxx_add_user_preferences.py`

### Privacy & Purge
- [ ] T144 Privacy purge endpoint `app/api/privacy_endpoints.py`
- [ ] T145 Privacy purge service with cascade deletion (favorites, history) `app/services/privacy_purge_service.py`
- [ ] T146 Unit test purge cascade `tests/unit/test_privacy_purge_cascade.py`

### Phrase Suggestions Model & Migration
- [ ] T147 Alembic migration phrase_suggestions table `alembic/versions/xxxx_create_phrase_suggestions.py`
- [ ] T148 [P] PhraseSuggestion model `app/models/phrase_suggestion.py`

### Metrics Instrumentation (FR-011)
- [ ] T149 Implement translation latency instrumentation (segment timers) `app/core/metrics_translation.py`
- [ ] T150 [P] Implement POI & phrase suggestion metrics (latency, cache hit) `app/core/metrics_navigation_phrase.py`

### Field Deprecation (FR-012)
- [ ] T151 Field deprecation dual mapping utility `app/core/deprecation.py`
- [ ] T152 Unit tests for deprecation mapping `tests/unit/test_field_deprecation.py`

**Note**: New tasks extend numbering beyond original 139 → total now 152.
