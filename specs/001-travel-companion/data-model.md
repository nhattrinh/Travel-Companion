# Data Model Overview (Draft)

Entity summaries mapped from `spec.md` for early scaffolding.

| Entity | Key Fields | Notes |
|--------|------------|-------|
| User | id, email, hashed_password, preferences (JSONB), created_at | Preferences store language pairs & privacy flags |
| TranslationItem | id, source_text, target_text, confidence, timestamp, source_type, context_tags, trip_id (nullable) | History + trip filtering |
| Phrase | id, canonical_text, translations JSON, phonetic, context_category | Suggestion ranking features later |
| PhraseSuggestion | id, phrase_id, relevance_score, triggering_context, recency_features JSON | Derived scoring inputs |
| FavoriteItem | id, user_id, target_type, target_id, created_at | Polymorphic reference |
| Place | id, name, category, lat, lon, etiquette_notes, favorite_flag | Cached POIs |
| Trip | id, user_id, started_at, ended_at (nullable) | Links history scope |
| MetricsRecord | id, metric_type, value, recorded_at | Aggregation or time series store |

Migrations will be created in later tasks (Phase 2+). This file will evolve as models gain fields.
