# Local reliability fixes — 2026-10-01

This batch implements the immediate fixes from the repository audit. The product remains a local, single-user CSV analytics application. Account authentication was not added.

## Implemented

- Removed generated Python execution and its helper. AI produces a validated chart plan or a clarification; manual and AI charts use the same aggregation service.
- Preserved missing observations and detached source data. Cleaning no longer imputes values automatically. Conservative parsing retains mixed or ambiguous strings and leading-zero identifiers.
- Made normalized schema keys deterministic and collision-free. AI enrichment can suggest compatible semantic types without renaming schema keys or converting text into metrics.
- Aggregated source rows exactly once. Date buckets retain their original column identity, count counts non-null observations, mean ranks by mean, null groups remain present, all-null sums remain null, and synthetic Others labels avoid collisions.
- Validated chart operations, filter operands, source columns, request limits, and AI plans. Optional provider clients initialize lazily with bounded timeouts; keyless startup, upload, and manual charting work.
- Added locks around in-memory dataset lifecycle operations and moved blocking routes to the server threadpool.
- Hardened frontend request freshness, dataset switching, browser persistence, history restoration, effective filter composition, date drilldown, empty responses, and consistent numeric/percentage formatting. Export code loads when needed.

## Architecture tradeoffs

Removing arbitrary code execution narrows AI requests to supported grouped charts. Predictions, derived fields, and unsupported statistics return a clarification rather than executing model-written code. This is a deliberately small boundary; a real sandbox would be a separate project.

Keeping source and parsed frames uses additional memory but avoids losing observations during cleaning. Storage still expires in memory after an hour of inactivity, evicts at ten datasets, and disappears when the backend restarts. Browser history is not durable dataset storage.

## Validation

Backend regression tests use synthetic data and mocked providers. No live AI provider calls are required or made. The historical audit probes were adapted to the corrected helper contracts; the assertions in backend/tests are the maintained regression suite.

Verified on the final changes:

- Backend: 50 unittest regressions passed with OPENAI_API_KEY disabled.
- Frontend: ESLint, TypeScript, helper assertions, and production build passed.
- Independent Luna xhigh review: no remaining findings after CSV-token, null drilldown, filter-provenance, and timezone fixes.
- Live API: offline upload/profile/manual aggregation, leading-zero IDs, null serialization, invalid filter/operation/limit rejection, optional AI 503, and date count/bucket drilldown passed.
- Production browser: keyless Gapminder demo, manual average chart, pinning, Europe filter with expected 71.9 mean, source-row drilldown, actual CSV upload, 42.5% ratio average, literal NA category, persisted dataset reload, history restore, and dataset switch without stale chart/filter/history passed.
- Browser console: no application errors. Recharts emits transient container-size warnings during chart mounts; this remains a minor polish item.
- git diff --check passed. No live provider calls were made.

The first browser attempt with the Next development server encountered local file-watcher limits. Browser verification used the production build. Use the commands in README.md to reproduce the automated checks.

## Follow-up decisions

Durable local storage and installation/packaging, dependency/security upgrades, larger-file resource limits, explicit AI data-sharing controls, and model evaluation remain separate work. The AI options audit records candidates; this batch does not introduce a new provider or claim production readiness.
