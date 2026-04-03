# Backend Architecture Walkthrough

This refactor keeps endpoint behavior unchanged while splitting `backend/main.py` into a thin entrypoint plus focused modules.

## Module Boundaries
- `backend/main.py`: app creation, middleware, router registration, uvicorn bootstrap.
- `backend/routers/*`: HTTP route definitions only.
- `backend/services/*`: orchestration/business flow for ingestion, aggregation, and query logic.
- `backend/utils/*`: shared helpers (filtering, CSV/DataFrame helpers, safe exec, error mapping, timing logs).
- `backend/core/config.py`: app constants, shared OpenAI client, prompts, and demo dataset config.

## Request Flow
- Ingestion (`/upload`, `/load-demo`): route validates input and source, service runs cleaning/profile/suggestions/store pipeline, response builder shapes `UploadResponse`.
- Aggregation (`/aggregate`): route delegates to aggregation service, which handles filtering, semantic enforcement, smart grouping/resampling, optional analysis, and `ChartResponse`.
- Query (`/query`): route delegates to query service, which performs LLM routing (JSON config or Python tool path), filtering, validation, aggregation, and response shaping.
- Drilldown (`/drilldown`): route uses shared filter utility and returns raw row slices.

## Why This Is Better For Review
- Route files are short and readable.
- Business logic is grouped by behavior, not mixed with app bootstrap.
- Shared utilities remove duplication and make testing targeted.
- Refactor is structural: API contracts remain unchanged.
