# Analytico production-readiness audit — October 1, 2026

> Historical pre-fix audit. Findings, test counts, dependency counts, and line references below describe the original checkout, not the current implementation. See [completed reliability fixes](local-safety-fixes-2026-10-01.md) for resolved items and [dependency audit](dependency-audit-2026-10-01.md) for current package status. Account authentication was excluded from the local-only scope. The adapted probe script now reports current behavior; maintained regression assertions live in backend/tests.

Analytico has a coherent exploration/dashboard workflow and a manageable codebase. It is not ready for an internet-facing, multi-user production deployment. The largest problems are unsafe execution, silent changes to data and query meaning, ephemeral shared state, and unbounded synchronous work. These are more consequential than the choice of AI model.

This is a review and decision document; application code has not been changed. Rankings combine potential harm, ordinary-user exposure, and how silently the failure occurs. They are priorities, not estimates of engineering effort. P0 means a release blocker for public deployment; P1 means fix before relying on the product for business decisions; P2 means reliability/performance/product work after those blockers.

## Scope and evidence

Reviewed first-party backend modules, routes, schemas, storage, helpers and tests; frontend state, requests, charts, filters, history, dashboard, export, styling and shared UI; dependency manifests, tracked-file inventory, and architecture/numeric-parsing documentation. Dependency implementations, an external deployment, real customer data, browser/device behavior and provider performance were not exhaustively audited.

| Check | Result | Interpretation |
| --- | --- | --- |
| Backend unittest discovery | 14/14 pass; also rerun with AI disabled/mocked | The current tests do not establish aggregate/query correctness. |
| TypeScript `npx tsc --noEmit` | Pass | Static types do not validate runtime backend/model output. |
| Frontend `npm run build` | Pass | Does not gate on the ESLint failures below. |
| Frontend `npm run lint` | 16 errors, 4 warnings | Includes conditional hooks, explicit `any`, and render purity errors. |
| `npm audit --json` | 47 affected package entries: 2 critical, 29 high, 15 moderate, 1 low | Includes transitive/dev packages; not 47 proven exploitable application paths. |
| Synthetic audit probes | Confirmed aggregation, parsing, filtering and sandbox defects | See [repro script](audit_probes.py); uses synthetic data and its own temporary CSV. |
| Local taxi demo benchmark, AI disabled | 1,068,755 rows × 20 columns; CSV parse 3.559 s; clean 7.140 s; cleaned frame 198.8 MiB; process peak RSS 1,300.1 MiB | Single local run, not a load test or a production p95. Excludes AI, full response construction and charting. |

Initial tests with a placeholder key attempted schema-enrichment calls and fell back after authentication failure. The offline rerun passed all 14 tests without provider calls. No valid-key model benchmark was performed. A clean-start probe with dotenv disabled confirmed import failure without an AI key.

## Ranked findings

| Rank | Priority | Fragile area | Main consequence | Evidence |
| --- | --- | --- | --- | --- |
| 1 | P0 | Generated Python executes in the API process | File access and unbounded computation under server privileges | Reproduced harmless file read/write |
| 2 | P1 | Aggregation occurs more than once | Incorrect counts, ranking and result semantics | Reproduced |
| 3 | P1 | Parsing and imputation overwrite source data | Plausible charts based on altered or lost observations | Reproduced |
| 4 | P0 | No authentication or dataset ownership | Dataset IDs act as bearer access; public compute/provider abuse | Code |
| 5 | P1 | Charts lack a durable executable query identity | Refresh, analysis and drilldown change the question | Code + API probe |
| 6 | P1 | Blocking work and missing resource budgets | One upload/AI/code request can stall a worker or exhaust memory | Code + local measurement |
| 7 | P1 | Process-local global dataset storage | Restart, worker routing and capacity eviction lose data | Code |
| 8 | P1 | Outdated dependencies and no update gate | Known affected versions can reach deployment | Registry audit |
| 9 | P1 | AI classifications override physical data reality | Failed ingestion, nondeterministic schema, invalid measures | Reproduced + code |
| 10 | P1 | Insights are produced without computed evidence | Unsupported claims appear as business analysis | Mocked query + code |
| 11 | P1 | Loose request/model/filter contracts | Invalid operations accepted, constraints silently dropped | Reproduced |
| 12 | P1 | Data disclosure has no product-level controls | Uploaded values leave the app automatically | Code |
| 13 | P1 | Frontend request races and cross-dataset history | Wrong chart/dataset/filter combinations | Code; browser timing untested |
| 14 | P1 | Conditional React hooks | Runtime crashes on certain chart/answer transitions | ESLint + code |
| 15 | P1 | Units, currency, percentage and measure semantics | Correct numbers displayed with incorrect meaning | Code + demo |
| 16 | P1 | Tests omit the dangerous paths | Regressions pass build and existing suite | Test inventory |
| 17 | P1 | Deployment and observability gaps | Hard to reproduce, detect and diagnose production failures | Tracked inventory + code |
| 18 | P2 | Browser-only persistence is fragile and unbounded | Lost work, stale sensitive snapshots, quota failures | Code |
| 19 | P2 | Export captures the UI in one large bitmap | Clipped charts, large memory spikes, weak report provenance | Code; visual output untested |
| 20 | P2 | Filtering/settings interface covers only samples | Users cannot express supported queries reliably | Code |
| 21 | P2 | Client rendering/bundling scales poorly | Heavy initial load and costly dashboard interactions | Code; no browser profiler trace |
| 22 | P2 | Accessibility and presentation defects | Keyboard/screen-reader friction and inconsistent styling | Static review |

### 1. Generated Python is not sandboxed

[Execution helper](/Users/rayan/.gemini/antigravity/scratch/analytico/backend/utils/execution.py:13), [query tool path](/Users/rayan/.gemini/antigravity/scratch/analytico/backend/services/query_service.py:104).

The docstring says AST whitelisting, but the implementation bans a few imports/calls and exposes full Pandas and NumPy modules. `df.to_csv(path)` and `pd.read_csv(path)` pass and execute; the probe only used an isolated temporary fixture. Empty builtins do not remove the capabilities of supplied Python objects. There is no execution deadline, memory/process isolation or output budget. Exceptions become strings that can be returned as apparently successful analysis; `print` is suggested by the tool contract but unavailable in the execution globals.

**Fix:** disable this path for public release. Use a validated operation plan for the ordinary analytics surface. If arbitrary advanced code is a deliberate product feature, run it in a disposable isolated process/container with no credentials, denied network, explicit mounted data, CPU/memory/time/output limits and a typed result/error contract. Strengthening the AST blacklist is insufficient.

### 2. The aggregation pipeline returns wrong answers

[Aggregation helpers](/Users/rayan/.gemini/antigravity/scratch/analytico/backend/modules/aggregation.py:12), [manual orchestration](/Users/rayan/.gemini/antigravity/scratch/analytico/backend/services/aggregation_service.py:30), [AI orchestration](/Users/rayan/.gemini/antigravity/scratch/analytico/backend/services/query_service.py:272).

Top-N grouping and date resampling already aggregate, then `aggregate_data` aggregates again. Counting groups of 2 and 4 yields 1 and 1. Counting 120 daily rows yields a total of 18 after weekly resampling. Other problems:

- With Others enabled, ranking always uses sum, including for requested mean/count/median. Ten rows averaging 1 outrank a category averaging 9 in the reproduced mean case.
- The manual path passes the original requested aggregation into top-N after semantic rules have changed the effective aggregation. Different stages can execute different operations.
- Labels become strings and sort lexicographically; numeric labels `1, 10, 2` no longer have numeric order. Time results can be sorted by value by the manual default.
- The final helper truncates before the manual orchestration sorts by value. The ordinary category top-N path sometimes masks this; date/high-cardinality paths can still select the wrong subset. AI `sort_by` is unused.
- Pandas grouping drops null grouping keys; all-null sums become zero; response shaping also changes NaNs to zero. Unknown and observed zero become indistinguishable.
- A real category called `Others` can collide with the synthetic bucket; helper names such as `_x_grouped`/`_period` can overwrite user columns.

**Fix:** operate on raw selected rows, assign grouping keys once, aggregate once, rank by the requested measure, then limit. Define record count versus non-null measure count, unknown-group policy, empty/all-null behavior and Others semantics explicitly. Keep chronological/numeric sort types. Reconcile totals and counts against an independent reference on fixtures.

### 3. Ingestion destroys information and imputes without a statistical basis

[Numeric/date conversion](/Users/rayan/.gemini/antigravity/scratch/analytico/backend/modules/data_janitor.py:344), [imputation](/Users/rayan/.gemini/antigravity/scratch/analytico/backend/modules/data_janitor.py:448), [CSV inference](/Users/rayan/.gemini/antigravity/scratch/analytico/backend/utils/dataframe_utils.py:18).

Only the changed DataFrame is retained. A numeric-looking majority in the first 100 non-null rows allows destructive full-column coercion; `['1','2','bad']` becomes `[1,2,null]`. `1.234,56` becomes `1.23456`; `001` becomes `1`, and a missing coded value becomes `-1`. CSV dtype inference can remove identifier formatting even before the janitor. Date parsing chooses a format from early rows, coerces failures, and has no user-selected locale/timezone policy. Ambiguous day/month values can pass with a wrong interpretation.

Automatic mean/median filling changes totals, distributions and correlations. The sensor fixture's observed total is 26.6; two filled values add approximately 8.8666. A bounded ratio `[0.2,0.8,null]` fills with `0.2` despite equally frequent modes. Density, integer-likeness and an AI `metric` label do not establish that an estimate is appropriate for descriptive analytics. Missing counts are captured after parsing, so existing nulls and parse failures are mixed.

**Fix:** immutable raw data; parsed views with explicit locale/unit policy; preserve missingness and parse-failure masks. Leave missing observations missing by default. Offer a separately selected, versioned transformation with affected cells, method and assumptions visible. Show observed and estimated results separately. For predictive work, benchmark an imputer on held-out data and avoid fitting preprocessing on validation/test data.

The existing [numeric-parsing plan](numeric-parsing-improvement-plan.md) correctly proposes raw/parsed separation, but the implementation does not provide it yet. A rightmost-separator heuristic alone remains ambiguous for values such as `1,234`; prefer explicit locale plus validation.

### 4. There is no access-control model

[Route registration](/Users/rayan/.gemini/antigravity/scratch/analytico/backend/main.py:27), [lookup](/Users/rayan/.gemini/antigravity/scratch/analytico/backend/storage.py:65), [drilldown](/Users/rayan/.gemini/antigravity/scratch/analytico/backend/routers/analytics.py:25).

Routes have no authentication/authorization dependency; datasets have no owner or tenant. Anyone holding a valid dataset ID can query or retrieve rows. UUIDs make guessing difficult but do not authorize access. CORS only restricts certain browser requests, not direct HTTP clients. Upload, demo loading and paid model calls have no per-user quotas. There is no deletion endpoint.

**Fix:** authenticate requests and enforce dataset ownership on every operation; apply per-account storage/compute/model budgets and retention/deletion rules. For a local-only product, deliberately constrain deployment scope; that still does not fix unsafe execution or correctness.

### 5. A chart response is being used as its own query plan

[Auto-refresh and analysis](/Users/rayan/.gemini/antigravity/scratch/analytico/frontend/src/app/page.tsx:43), [drilldown](/Users/rayan/.gemini/antigravity/scratch/analytico/frontend/src/components/SmartChart.tsx:157), [filter skipping](/Users/rayan/.gemini/antigravity/scratch/analytico/backend/utils/filtering.py:100).

Refresh and Analyze reconstruct a manual request from rendered axes and current global settings. They omit the AI filters that created the chart. A request for revenue above a threshold can become unfiltered when the user changes Top N or asks for analysis. Resampled axes such as `pickup_date_week` do not exist in the stored source DataFrame, so refresh validation fails. Drilldown silently skips that synthetic-column filter and can return rows from every time bucket. The API probe confirmed a nonexistent filter column returns all three rows with HTTP 200. Python-derived result columns are similarly not reproducible from raw axis names.

Pinned widgets store results, not the complete executable plan; widget drilldown combines snapshot AI filters with today's global filters. The backend's `llm_filters` actually contains both global and AI filters, and the frontend adds global filters again.

**Fix:** persist a validated `AnalysisPlan` and its ID/version separately from the result. Include source column IDs, all filters, transformations, time bucket/timezone, measure definitions, order/limit and dataset version. Refresh executes the same plan with deliberate edits; Analyze describes that result; drilldown uses the plan plus the clicked bucket's source-row predicate. Reject unknown filter columns.

### 6. Blocking work and unbounded inputs threaten availability

[Async routes invoking sync work](/Users/rayan/.gemini/antigravity/scratch/analytico/backend/routers/analytics.py:15), [upload route](/Users/rayan/.gemini/antigravity/scratch/analytico/backend/routers/ingestion.py:30), [sync AI clients](/Users/rayan/.gemini/antigravity/scratch/analytico/backend/core/config.py:24).

Async endpoints perform CSV parsing, Pandas work, synchronous network calls and generated execution directly on the event loop. The measured upload preparation occupied roughly 10.7 seconds before optional AI work. There are no application upload-byte/row/column/cell budgets, query deadlines, concurrency limits or per-user queues. DataFrame copies, full unique arrays and row-wise Python conversion multiply memory costs. Ten dataset slots are not a memory budget. Model calls inherit SDK defaults rather than an explicit application deadline/retry budget; Axios has no timeout or cancellation policy.

**Fix:** enforce budgets before expensive work, limit concurrent ingestion/queries, use asynchronous provider requests, and offload bounded Pandas operations. Threads are a tactical way to free the event loop; CPU-heavy/large jobs need bounded worker processes and job status/cancellation. Move schema summaries off the critical upload path. Measure queue time, peak memory and concurrent p95, not only one demo's duration.

### 7. Storage cannot support reliable multi-worker production

[Global storage and eviction](/Users/rayan/.gemini/antigravity/scratch/analytico/backend/storage.py:54).

The entire backend state is a process-local dictionary. Restarting loses everything; multiple workers disagree about dataset IDs; the eleventh upload evicts another dataset regardless of ownership. TTL cleanup only runs during later requests, so idle expired data can remain allocated. There is no source version/content identity, durable dashboard store, recovery or atomic ingest publication. Ingestion stores the dataset before response construction is validated, allowing failed responses to leave orphaned allocations.

**Fix:** durable raw/parsed columnar files with metadata/ownership/versioning in a database. Cache working frames by byte budget as an optimization. Publish ingestion state only after validation; model enrichment can finish later. Make expiration/deletion explicit and test process restart and multiple workers. Avoid putting all large frames into Redis by default; metadata and result caching are different workloads.

### 8. Dependency remediation needs a release gate

[Frontend manifest](/Users/rayan/.gemini/antigravity/scratch/analytico/frontend/package.json:12), [backend requirements](/Users/rayan/.gemini/antigravity/scratch/analytico/backend/requirements.txt:1).

The local npm audit reported critical entries for Next.js and jsPDF, high for Axios and numerous indirect packages. This is a package-version finding, not proof that every advisory is reachable. For example, the jsPDF file-inclusion issue targets its Node build, whereas this application exports in the browser; Next.js image/RSC/Windows-specific advisories have deployment prerequisites. Nevertheless, pinned Next 16.1.3 and installed dependency versions need review before public exposure. See the [Next.js AVIF advisory](https://github.com/advisories/GHSA-2xp9-vwfh-vxw4) and [jsPDF file-inclusion advisory](https://github.com/advisories/GHSA-f8cm-6447-x5h2).

Backend dependencies use only lower bounds and no lockfile, so a fresh installation can change behavior substantially. `httpx`, used by FastAPI TestClient, is not explicitly declared as a test dependency.

**Fix:** select supported patched versions after checking advisory prerequisites and migration notes, regenerate a reproducible lock, run meaningful regression checks, and add routine dependency/security review. Do not blindly run a forced audit fix: several reported remediations are outside current ranges or unavailable to that command. Add explicit development/test dependencies and a Python lock/runtime version.

### 9. AI schema interpretation can contradict the data

[Enrichment and header handling](/Users/rayan/.gemini/antigravity/scratch/analytico/backend/modules/data_janitor.py:88), [semantic precedence](/Users/rayan/.gemini/antigravity/scratch/analytico/backend/services/ingestion_service.py:54), [profiling](/Users/rayan/.gemini/antigravity/scratch/analytico/backend/modules/intelligence.py:113).

An AI `metric` classification overrides the deterministic detector without checking whether the series is numeric. A mocked `metric` label for `['142 min','200 min']` causes profiling to fail converting `'142 min200 min'` to float. AI-generated names become storage keys, so availability/model changes can alter schema identity between uploads. Deduplication does not reserve already suffixed names: unique raw headers `A!`, `A?`, `a_2` normalize to `a`, `a_2`, `a_2` and crash column processing. Substring rules also misclassify names, and bounded numeric values are labeled percentages without units.

**Fix:** stable internal column IDs and a lossless original-name map; AI proposes display aliases and semantic roles. Validate roles against physical dtype/parsing evidence, allow unknown/ambiguous, expose corrections, and persist decisions by dataset version. Make deduplication globally collision-free. Never let AI labels alone authorize a numerical transformation.

### 10. Analysis text can claim facts the engine never computed

[Query prompt/result](/Users/rayan/.gemini/antigravity/scratch/analytico/backend/services/query_service.py:31), [default analysis](/Users/rayan/.gemini/antigravity/scratch/analytico/backend/modules/intelligence.py:62), [calculated-field promise](/Users/rayan/.gemini/antigravity/scratch/analytico/backend/core/config.py:50).

The planning call sees five sample rows and returns the final chart's analysis before full aggregation/filtering. The server copies that analysis directly into the result. Default charts attach canned trend/seasonality/value language; a composed sum chart is labeled correlation without calculating correlation. The prompt advertises a `calculated_field` that the query executor does not implement. Queries have no conversation history despite a chat-like interface. Malformed JSON becomes a successful text answer; invalid tool execution can also surface as a successful answer.

**Fix:** separate planning, deterministic execution and grounded explanation. Feed the explanation the actual result, sample coverage, denominator/units and warnings. Support only implemented capabilities, and request clarification for ambiguous questions. Measure factual claims against computed quantities. Structured output controls shape, not truth.

### 11. Invalid plans and filters fail open

[Request models](/Users/rayan/.gemini/antigravity/scratch/analytico/backend/models.py:70), [filtering](/Users/rayan/.gemini/antigravity/scratch/analytico/backend/utils/filtering.py:22), [query parsing](/Users/rayan/.gemini/antigravity/scratch/analytico/backend/services/query_service.py:238).

Aggregation/chart/operator/sort strings lack enums. Lists and prompt lengths are unbounded. Negative drilldown limits are accepted (`-1` returns all but the final row). An invalid aggregation silently executes sum but can still be labeled with the invalid string. Bad AI filters are discarded, permitting an unfiltered chart. Unknown filter columns are ignored; malformed numeric/date filters can be skipped. Equality uses string comparisons, so numeric `1.0 == 1` fails. `/aggregate` with explicit `sort_by:null` crashes because `DatasetInfo` has no `columns` property. The frontend preview helper points at an endpoint that does not exist, though it is currently unused.

**Fix:** strict Pydantic enums/discriminated filters, bounded lengths/limits, typed equality/coercion, dtype/operator compatibility and validation of the whole plan before execution. Reject unsupported or dropped constraints explicitly. Generate or validate frontend contracts from OpenAPI and use typed error/status codes. Treat NaN/infinity/date/Series output serialization consistently; hide internal stack/provider error details.

### 12. External data use is automatic and difficult to control

[Column samples](/Users/rayan/.gemini/antigravity/scratch/analytico/backend/modules/data_janitor.py:108), [upload summary](/Users/rayan/.gemini/antigravity/scratch/analytico/backend/services/ingestion_service.py:22), [query sample](/Users/rayan/.gemini/antigravity/scratch/analytico/backend/services/query_service.py:35).

Schema enrichment sends three values per column; upload summaries send full first rows; queries send another sample. There is no product setting for external AI use, minimization/redaction, field sensitivity or provider routing. Sample cell length is not bounded on ingestion prompts. Uploaded text can also contain instructions that influence interpretation; it is not trustworthy instruction content. Clear Data removes the active pointer but does not erase retained history/dashboard snapshots.

**Fix:** explicit dataset/workspace AI policy and minimal metadata payloads; omit sensitive values by default; field-level redaction, prompt-size limits and deletion semantics covering backend and browser artifacts. Keep provider policies/versioning visible. Treat cell content as data and enforce tool permissions independently of prompts.

### 13. Request/state ownership is missing in the frontend

[History/state](/Users/rayan/.gemini/antigravity/scratch/analytico/frontend/src/context/DataContext.tsx:199), [refresh effect](/Users/rayan/.gemini/antigravity/scratch/analytico/frontend/src/app/page.tsx:43), [upload/default chart](/Users/rayan/.gemini/antigravity/scratch/analytico/frontend/src/components/FileUploader.tsx:43).

Async requests write directly to global state without cancellation or checking that the dataset/request is still current. A slower earlier filter response can overwrite a newer one; an in-flight query can repopulate charts after Clear/Upload New. History records contain no dataset ID, and switching datasets does not clear or scope history. Selecting an old chart can display it using new dataset formatting and send drilldown/analysis against the new dataset. Restoring an old dataset after an async validity check can race with new user work. Transient backend errors are treated as expired datasets by `validateDataset`.

**Fix:** request IDs/generations, cancellation and dataset-version checks before commits to state. Scope history and plans by dataset/workspace. Distinguish unavailable backend from actual 404/expiration. Derive busy state from active operations instead of one shared boolean. Reset dependent builder/table/drilldown state deliberately.

### 14. Chart render transitions can violate React's hook rules

[Early returns](/Users/rayan/.gemini/antigravity/scratch/analytico/frontend/src/components/SmartChart.tsx:106), [conditional hooks](/Users/rayan/.gemini/antigravity/scratch/analytico/frontend/src/components/SmartChart.tsx:203).

Three `useMemo` calls appear after empty/text-answer early returns. Changing between those branches in the same mounted chart changes the hook count. ESLint confirms the rule violation; a browser reproduction was not run. Normal query loading sometimes unmounts the chart, which can mask the failure, but history transitions need not. Clarification responses with only `analysis` return no chart content because they lack `answer`; the explanation is only available through the separate toggled panel.

**Fix:** put unconditional hooks before branch returns or split chart/text/empty states into separate renderers. Make clarification a typed, visible state and do not toast it as a successfully generated chart. Correct render purity and enforce lint in CI.

### 15. Measure semantics and display metadata are too weak

[Format inference](/Users/rayan/.gemini/antigravity/scratch/analytico/backend/modules/data_janitor.py:188), [chart formatter](/Users/rayan/.gemini/antigravity/scratch/analytico/frontend/src/components/SmartChart.tsx:32), [table formatter](/Users/rayan/.gemini/antigravity/scratch/analytico/frontend/src/components/DataTable.tsx:37).

Every currency renders as USD, including rupee/euro parsing. Charts multiply percentage decimals by 100 while the data table does not: `0.2` can be 20% in the chart and 0.2% in the table. Count results reuse a source field's currency/percentage format. All series share one Y scale/primary format even when units differ. The offline Gapminder default sums population and life expectancy by year on the same axis, despite life expectancy being non-additive. Semantic rules change every selected measure to count if any selected field is categorical/identifier.

**Fix:** explicit measure definitions: physical type, unit/currency, additive/non-additive behavior, default aggregation, missing policy and output format. Format aggregate results from output metadata, using one shared formatter. Validate chart compatibility, use separate axes when meaningful, and support weighted means only with an explicit weight. Do not silently choose sum for every numeric field.

### 16. Verification is missing exactly where correctness matters

[Backend tests](/Users/rayan/.gemini/antigravity/scratch/analytico/backend/tests/test_api_smoke.py:85), [scripts](/Users/rayan/.gemini/antigravity/scratch/analytico/frontend/package.json:5).

Smoke tests mock ingestion and primarily exercise route reachability. There are no aggregate/query execution, unsafe-code, storage lifecycle, AI contract/evaluation or frontend workflow tests. Some cleaning tests assert statistical filling as desirable without validating the consequences. Provider behavior can leak into tests when an API key exists. Build and type-check pass while numerical/hook failures remain.

**Fix:** a small high-value fixture suite, not snapshots of implementation details: counts across bins, weighted/non-additive measures, null groups, top-N conservation, parsing loss/locale/identifiers, invalid filters/plans, retries/refusals, ownership, restart/multi-worker persistence, filter→refresh→Analyze→drilldown parity and stale responses. Mock providers in deterministic CI; maintain separate opt-in live evals.

### 17. Operational readiness is absent from the repository

[Configuration](/Users/rayan/.gemini/antigravity/scratch/analytico/backend/core/config.py:24), [timing logs](/Users/rayan/.gemini/antigravity/scratch/analytico/backend/utils/pipeline_logging.py:4), [health](/Users/rayan/.gemini/antigravity/scratch/analytico/backend/routers/health.py:8).

Backend startup requires an OpenAI key even for manual analytics. CORS origins are development-only and the client defaults to localhost. The advertised taxi demo is ignored/untracked with no provisioning/download instructions. There is no tracked CI/deployment configuration, runtime pin, readiness check, backup/recovery procedure or observability setup. The root health route does not test useful dependencies. Timing reports CSV ingestion as effectively zero because parsing happened before its timer starts; stdout logs have no correlation IDs, structured status or token/cost/queue metrics.

**Fix:** validated environment configuration and optional lazy AI initialization; reproducible packaging/provisioning, CI gates, deployment/readiness configuration and graceful shutdown. Structured request/job/model telemetry with safe logs and explicit error classes; track parse, queue, execution and provider time separately. Deployment-specific infrastructure may exist elsewhere, but the repo provides no evidence of it.

### 18–22. Product resilience and performance follow-ups

**18. Browser persistence:** [DataContext](/Users/rayan/.gemini/antigravity/scratch/analytico/frontend/src/context/DataContext.tsx:161) parses saved objects without schema migration/validation, writes without quota/error handling, and retains dashboard stores for old ephemeral dataset IDs indefinitely. A delayed write can be lost on close; a changed layout rewrites the entire store. A backend restart invalidates dataset access but leaves saved snapshots. Persist plans/dashboards server-side with versions; treat local storage as a bounded cache and show save failures. Keep sensitive retained data and Clear/Delete behavior understandable.

**19. Exports:** [exportReport](/Users/rayan/.gemini/antigravity/scratch/analytico/frontend/src/lib/exportReport.ts:86) creates one whole-dashboard canvas at 2× pixel ratio before splitting it. Memory scales with the entire surface, not a page. Charts containing more than 12 points render horizontally scrolling surfaces; screenshots can clip hidden points. Page breaks chosen from widget bottoms can cut another widget spanning that break. Reports omit executable query/dataset-version/cleaning provenance; controls in dashboard headers are not marked for exclusion. Prefer bounded per-widget/page rendering with explicit content/layout and metadata. Visual QA across small/large dashboards and long labels is required before promising report fidelity. No exported PDF was visually verified during this audit.

**20. Filtering:** [FilterBar](/Users/rayan/.gemini/antigravity/scratch/analytico/frontend/src/components/FilterBar.tsx:144) hides all settings if no categorical fields exist, exposes only six categorical columns and the first 20 sample values, and has no numeric/date/operator UI despite backend support. First-observed samples are not a full domain or a representative set. “Show All” still caps/groups at 500. Provide paginated/searchable distinct values and typed numeric/date filters; expose settings independently; name the cap accurately.

**21. Client performance:** [page](/Users/rayan/.gemini/antigravity/scratch/analytico/frontend/src/app/page.tsx:19) imports chart/dashboard/export functionality into the initial client tree. One context value exposes all state and changes on every provider render; memoized tooltip work does not isolate dashboard charts from global updates. A 500-point chart can be 25,000 CSS pixels wide with SVG marks/dots. Dashboard layout callbacks update and persist snapshots on every change and overwrite one canonical layout using the current responsive layout. Dynamically load export/dashboard/builder paths, split subscriptions by responsibility, preserve per-breakpoint layouts, update on settled interactions, and measure rendering/bundle size before tuning.

**22. Accessibility/presentation:** [drilldown](/Users/rayan/.gemini/antigravity/scratch/analytico/frontend/src/components/DrillDownModal.tsx:16) and [cleaning drawer](/Users/rayan/.gemini/antigravity/scratch/analytico/frontend/src/components/CleaningReportDrawer.tsx:43) lack dialog semantics, focus trapping/restoration and Escape behavior. Several icon buttons/toggles lack accessible names/state. Charts need a keyboard-accessible drilldown and data-table alternative that retains units. CSS defines `oklch(...)` tokens but callers wrap them in `hsl(var(...))`, producing invalid color values in grids/toasts/placeholders. DataTable pagination is not reset when data changes, and filter settings' delayed listener is not fully canceled. Resolve these with standard accessible primitives and shared styling/formatting; test keyboard and narrow-screen flows.

## Recommended architecture, with tradeoffs

Keep the existing FastAPI/Next.js shape. It is small enough to repair without a wholesale rewrite. Four deep modules would concentrate the currently scattered invariants:

1. **Dataset module:** immutable raw artifact + parsed view + column IDs, units, parse/null masks, owner/version and explicit transformations. Extra storage is the cost of reproducibility and reversible cleaning.
2. **Analysis module:** validates an `AnalysisPlan`, executes it once, returns typed measures/results and source-row predicates. Manual charting and AI cross the same interface. Pandas is sufficient initially; DuckDB/columnar storage is an evidence-based scaling option, not an automatic requirement. Constrained SQL still needs security/resource controls.
3. **AI module:** configurable adapters for bounded schema/intent/planning/explanation operations, versioned prompts, strict output, deadlines, token/cost budgets and eval metadata. Add only seams with an actual varying provider/task. Separate explanation availability from deterministic analytics availability.
4. **Workspace module:** durable dataset-scoped plans/history/dashboards plus frontend request ownership. Charts are renderings of a plan/result; they do not define execution semantics.

This centralizes correctness without adding a large agent framework, microservice fleet or separate implementation per model. Isolated advanced-code jobs remain a distinct capability if the product truly needs them.

## Product decisions that have the most leverage

- **Trustworthy descriptive analytics:** default to observed data, expose completeness/parse failures, and make cleaning reversible. This is the strongest immediate product direction.
- **Predictive analytics:** forecasting/imputation should be explicit tasks with evaluation and uncertainty, separate from observed-data dashboards. Do not imply forecasting support because Python can execute arbitrary code.
- **Reproducible reports:** retain dataset version, measures, filters and lineage; refresh reports by replaying the plan. Show exactly which data and assumptions support each insight.
- **Ambiguity handling:** ask which date, unit, denominator or business measure the user intended. A visible clarification is better product behavior than an unsupported confident chart.

## AI choices, including Jev

See the separately sourced [AI options audit](ai-options-audit-2026-10-01.md) for version-specific details and tradeoffs.

Jev is a text-input typed decision model suitable for choosing among supplied options/classifications, not a numerical imputer or narrative generator. A useful pilot is intent/column-role selection with an explicit unknown option. Its own documented limitations include numerical precision, dates and adversarial content; confidence needs held-out calibration on Analytico. [TypeSafe limitations](https://docs.typesafe.ai/model-jaggedness/jev-1.13)

Benchmark current OpenAI candidates against the existing model rather than replacing a hardcoded string: focused tasks with Luna, broader planning with Sol, complex reasoning selectively with Astra. Current APIs/configuration differ, so migration must follow official guidance. [OpenAI catalog](https://developers.openai.com/api/docs/models), [migration guidance](https://developers.openai.com/api/docs/guides/latest-model)

Strict Structured Outputs is an immediate improvement to plan/schema contracts; it does not establish semantic correctness or execution safety. [OpenAI Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs)

For other tools, prioritize explicit dataframe contracts (ordinary validation/Pandera), deterministic columnar query execution and portable evaluations. TabPFN is an optional predictive experiment with hardware/licensing checks, not a default cleaning dependency. See the cited research note before adopting it.

## Repair sequence and exit criteria

| Stage | Work | Evidence required to advance |
| --- | --- | --- |
| 1: Safety and correctness | Disable unsafe exec; reject invalid plans/filters; aggregate once; preserve raw values and leave nulls by default; fix hooks; remediate dependency release blockers | Adversarial capability checks; reference fixtures with exact totals/counts; malformed inputs reject cleanly; lint/typecheck/build pass |
| 2: Reproducibility | Canonical plans, stable column IDs/units, ownership, durable storage, dataset-scoped frontend history, cancellation/request ownership | Same plan yields same answer after restart and across workers; refresh/Analyze/drilldown parity; stale responses cannot change the current dataset |
| 3: Operational capacity | Upload/memory/concurrency/model budgets, bounded workers/jobs, readiness/configuration, structured metrics and recovery | Representative concurrent workload meets chosen p95/memory/error/cost targets; controlled timeout/cancellation/failure recovery |
| 4: Product polish and AI selection | Grounded explanations/evals; Jev shadow pilot; measured model routing; full typed filtering; persistence/export/accessibility/bundle improvements | Held-out task/result accuracy, clarification quality and grounded-claim checks; report visual QA and keyboard/device checks |

Target p95, maximum dataset size, concurrency, retention, budget and supported analytics need to be chosen for the intended deployment. Set those numbers from user/workload goals and measurements; this audit does not invent an SLA.

The highest-value next implementation is the shared analysis-plan/execution path plus raw-data preservation. It resolves several correctness failures together and gives model upgrades a stable, testable interface.
