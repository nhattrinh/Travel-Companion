# Travel Companion Architecture Overview

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         iOS Client (SwiftUI)                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │
│  │  Auth    │  │Translation│  │Navigation│  │Phrasebook│        │
│  │  Views   │  │  Camera  │  │   Map    │  │  Views   │        │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘        │
│       │             │              │              │              │
│       └─────────────┴──────────────┴──────────────┘              │
│                          │                                       │
│                    APIClient (URLSession)                        │
│                    Keychain (Token Storage)                      │
└──────────────────────────┬──────────────────────────────────────┘
                           │ HTTPS/REST
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FastAPI Backend (Python)                      │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    API Endpoints                         │   │
│  │  /auth   /translation   /navigation   /phrases   /trips │   │
│  └────┬─────────────────────────────────────────────────────┘   │
│       │                                                          │
│  ┌────▼─────────────────────────────────────────────────────┐   │
│  │                  Service Layer                           │   │
│  │  Auth  │  OCR  │  Translation  │  Navigation  │  Trip    │   │
│  └────┬─────────────────────────────────────────────────────┘   │
│       │                                                          │
│  ┌────▼─────────────┐         ┌──────────────┐                  │
│  │  PostgreSQL DB   │         │  Redis Cache │                  │
│  │  (SQLAlchemy)    │         │  (Optional)  │                  │
│  └──────────────────┘         └──────────────┘                  │
└─────────────────────────────────────────────────────────────────┘
```

### Component Breakdown

#### iOS Application
- **SwiftUI Views**: Declarative UI with reactive state management
- **ViewModels**: MVVM pattern with `@Published` properties
- **APIClient**: Centralized HTTP client with envelope response parsing
- **KeychainTokenStore**: Secure token persistence (access/refresh tokens)
- **Network Monitor**: Connectivity detection for offline handling

#### Backend Services
- **FastAPI Application**: Async Python web framework
- **SQLAlchemy ORM**: Database models with async support
- **Alembic Migrations**: Database schema versioning (6 migrations created)
- **Pydantic Schemas**: Request/response validation
- **JWT Authentication**: Token-based auth with refresh flow

### Data Models

#### Core Entities

**User** → **Trip** → **Translation** (one-to-many relationships)

**Phrase** ← **Favorite** → User (polymorphic favorites)

**POI** (independent geospatial data)

### API Design - Envelope Pattern (FR-010)

All responses follow consistent envelope format:
```json
{
  "status": "ok" | "error",
  "data": <response_payload> | null,
  "error": null | <error_message>
}
```

### Performance Targets Achieved

| Endpoint | p95 Target | Status |
|----------|-----------|--------|
| Translation live-frame | ≤1.0s | ✅ Validated |
| POI fetch | ≤800ms | ✅ With caching |
| Phrase suggestions | ≤300ms | ✅ With caching |
| Trip summary | ≤500ms | ✅ Validated |

### Security Features

- JWT tokens (HS256) with access/refresh flow
- Password hashing (bcrypt/PBKDF2)
- Keychain token storage (iOS)
- Input validation via Pydantic
- Rate limiting middleware

### Data Retention

- Completed trips: 30-day retention with automatic purge
- CASCADE deletion of related translations
- Active/archived trips preserved

---

*Last Updated: November 14, 2025*
*Version: 1.0 - Phases 1-6 Complete (All User Stories Implemented)*
