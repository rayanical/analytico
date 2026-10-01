# Backend architecture

Analytico runs locally as a single FastAPI process. Manual analytics are deterministic and available without an AI key. The frontend is a Next.js application connecting to the local API.

## Modules

- `backend/main.py`: app creation, localhost CORS policy, router registration and loopback bootstrap.
- `backend/routers/*`: HTTP routes; blocking ingestion and analytics handlers run in the server threadpool.
- `backend/models.py`: validated request, response and AI-plan contracts.
- `backend/services/ingestion_service.py`: cleaning, profiling, optional AI summaries, response construction and storage.
- `backend/services/aggregation_service.py`: shared filtering, raw-row grouping/date bucketing, one aggregation, sorting, limiting and optional result-based explanation.
- `backend/services/query_service.py`: obtains and validates an AI chart plan or returns a clarification, then calls the same aggregation service.
- `backend/modules/*`: conservative parsing, semantic suggestions, profiling and deterministic aggregation helpers.
- `backend/utils/*`: typed filters, CSV/DataFrame helpers, errors and timing logs. Generated Python execution is removed.
- `backend/core/config.py`: constants, demo paths, provider model configuration and lazy optional AI client.
- `backend/storage.py`: locked in-memory dataset lifecycle, with separate source and parsed frames.

## Request flow

1. Upload/demo reads source values as strings, preserving literal NA tokens and leading zeros; only blank CSV fields are missing by default.
2. Cleaning creates a parsed view, deterministic column keys and metadata. Missing observations remain missing; ambiguous/mixed values remain text.
3. Ingestion builds the response successfully before storing the source and parsed frames.
4. Manual requests and validated AI plans enter the same executor. It rejects unknown columns/invalid operands and aggregates source rows exactly once.
5. Responses identify the source axis, time bucket, effective filters and synthetic Others label. AI-only filters remain distinct so UI refresh can preserve them.
6. Drilldown applies the displayed chart's filters plus the clicked category or date interval against the parsed frame. Explicit null membership selects missing groups.

Source preservation does not mean the drilldown route returns original CSV lexemes; it currently returns parsed rows. Stored source data is retained internally for future reversible transformations.

## Current limits

Datasets are process-local, expire after an hour of inactivity, and are evicted at ten entries. Restart loses the working datasets; browser history/dashboard snapshots are not durable dataset storage. Threads avoid blocking the event loop but do not provide worker memory limits or backend job cancellation. A provider key enables optional outbound AI requests; per-dataset data-sharing controls remain pending.

See [validation and remaining work](local-safety-fixes-2026-10-01.md). Jev integration, durable workspaces, packaging and CI are separate changes.
