# Multi-Modal AI Travel Companion Backend & iOS Client

FastAPI backend + SwiftUI iOS app delivering live camera translation overlay, context-aware navigation guidance (POIs + etiquette), smart phrase suggestions, favorites, and persistent trip memory. Includes privacy purge, metrics instrumentation, and deprecation utilities.

## High-Level Features

| Domain | Key Capabilities |
|--------|------------------|
| Translation | Live frame OCR + translation (p95 ≤1.0s), static image fallback, history save |
| Navigation | Nearby POIs, contextual etiquette hints, caching layer |
| Phrasebook | Context-based phrase suggestions, favorites toggle, suggestion caching |
| Trip Memory | Trip lifecycle (start/end), recent translations, favorites continuity, retention purge |
| Privacy | `/privacy/purge` endpoint + cascade deletion service |
| Metrics | Latency timers (translation, POI, phrase), system endpoint metrics, cache stats |
| Deprecation | Dual field mapping utility for graceful field evolution |
| iOS Client | Camera overlay, language selector, phrasebook, navigation map, trip overview |

## Repository Structure (Backend Extract)

```
app/
├── api/                       # Routers: auth, translation, navigation, phrasebook, trips, privacy, metrics
├── core/                      # db, security, jwt, logging, metrics, deprecation, validation
├── config/                    # Settings & loader
├── middleware/                # Request context, rate limit, auth
├── models/                    # SQLAlchemy models
├── schemas/                   # Pydantic schemas
├── services/                  # OCR, translation, navigation, phrase suggestions, trip, purge
└── ...

alembic/versions/              # Migrations
ios/TravelCompanion/            # SwiftUI app
specs/001-travel-companion/     # Feature spec, plan, tasks
```

## Environment & Configuration
Create `.env`:
```env
POSTGRES_URL=postgresql://user:pass@localhost:5432/travel
REDIS_URL=redis://localhost:6379/0
JWT_SECRET=changeme
LOG_LEVEL=INFO
DEBUG=true
```

## Backend Setup
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
alembic upgrade head
uvicorn app.main:app --reload
```
Docs: `http://localhost:8000/docs`

### Docker
```bash
docker compose -f docker-compose.travel.yml up --build
```

## iOS Setup
Open `ios/TravelCompanion/` in Xcode (Swift 5.9, iOS 17+). Configure base URL in `Shared/Config/Environment.swift`.

## Key Endpoints (Envelope `{status,data,error}`)
- `POST /translation/live-frame`
- `POST /translation/image`
- `POST /translation/save`
- `GET /navigation/pois`
- `GET /phrases` / `POST /phrases/{id}/favorite` / `GET /phrases/favorites`
- `POST /trips/start` / `POST /trips/{id}/end`
- `POST /privacy/purge`
- `GET /metrics`
- `GET /user/profile` / `PATCH /user/profile/preferences`

## Metrics & Instrumentation
Use timing context managers:
```python
from app.core.metrics_translation import record_translation_latency
with record_translation_latency():
   # pipeline
   pass
```
Snapshots include `p95_ms`, `p99_ms`, `count`.

## Deprecation Mapping
`DeprecationMapper` adds legacy field aliases (e.g. `segments_legacy`).
```python
mapper = DeprecationMapper({"oldField": "new_field"})
data = mapper.transform_outbound({"new_field": 1})
```

## Privacy & Retention
`POST /privacy/purge` clears translations, favorites, trips. Retention policy: ≤30 days for sensitive frame/location data (scheduler TBD).

## Testing
Run tests:
```bash
pytest -q
```
Categories: unit, integration, perf smoke (latency budgets).

## Performance Targets
- Translation overlay p95 ≤ 1000ms / p99 ≤ 1500ms
- Phrase suggestion p95 ≤ 300ms
- Navigation initial load p95 ≤ 800ms
- Favorites retrieval p95 ≤ 200ms

## Development Workflow
```bash
make dev-backend
make test
make lint
make migrate
```

## Roadmap
- Offline models
- ML ranking for suggestions
- Prometheus / OpenTelemetry export
- Extended deprecation audit reporting

## Security Notes
- Secrets only via env vars
- Validation in `core/validation.py`
- JWT auth (enhance for refresh/roles)

## Disclaimer
Pilot implementation; some advanced flows (full auth refresh, scheduler) are stubs.