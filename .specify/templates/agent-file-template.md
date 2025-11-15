# [PROJECT NAME] Development Guidelines

Auto-generated from all feature plans. Last updated: [DATE]

## Active Technologies

[EXTRACTED FROM ALL PLAN.MD FILES]

## Project Structure

```text
[ACTUAL STRUCTURE FROM PLANS]
```

## Commands

[ONLY COMMANDS FOR ACTIVE TECHNOLOGIES]

## Code Style

[LANGUAGE-SPECIFIC, ONLY FOR LANGUAGES IN USE]

## Recent Changes

[LAST 3 FEATURES AND WHAT THEY ADDED]

<!-- MANUAL ADDITIONS START -->
### Core Principles Snapshot (Auto-sync from Constitution)

1. Code Quality Discipline: Tests, coverage ≥80% on changed code, typed errors, clean diffs.
2. User Experience Consistency: Stable API envelope `{status,data,error}`, documented deprecations.
3. Performance & Efficiency Standards: p95 ≤200ms targets, profiling heavy operations, cache strategy.
4. Security & Reliability Constraints: Secret hygiene, validation, health checks, patch cadence.
5. Workflow & Quality Gates: CI stages (lint/tests/coverage/perf), reviewer confirmation, waiver governance.

### Enforcement & Workflow Notes
- All PR descriptions MUST cite impacted principles.
- Add profiling evidence for performance-sensitive changes (attach summary or link).
- Document any field deprecations in `docs/CHANGELOG.md`.
- Waivers expire in ≤30 days; track in a `WAIVERS.md` file if present.
- Templates (`plan`, `spec`, `tasks`, `checklist`) reflect principle alignment; do not fork them without governance approval.

<!-- MANUAL ADDITIONS END -->
