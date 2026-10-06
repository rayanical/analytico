# First milestone: dependable upload to answer

Status: first real-data iteration completed; release acceptance still in progress. Direction: [product goal](product-goal.md). Evidence and remaining coverage: [real-data scorecard](real-acceptance-2026-10-02.md).

## Outcome

A repeatable report demonstrates what a user experiences when importing unfamiliar data, inspecting it, requesting a chart, and using optional AI. The same report identifies the next highest-payoff fix and proves whether that fix improved the product.

This is a narrow vertical slice, not a replacement of DuckDB, a new UI framework, or a promotion of experimental AI batching. Reuse existing ingestion benchmarks, interpretation evaluations and deterministic analytics tests. Browser coverage supplies the missing user-facing evidence.

## Implementation sequence

1. Define a versioned fixture manifest with domain, dimensions, bytes, expected column interpretation, expected aggregates and expected clarifications. Include a public real dataset plus reproducible synthetic fixtures. Keep evaluation labels out of model prompts.
2. Cover small clean data, large data, wide data, and adversarial parsing/meaning cases: sales/returns, taxi measurements/codes, time series, leading-zero identifiers, locale decimals, mixed currencies, nulls, ambiguous dates and malformed late rows. Use independent holdout cases to discourage tuning solely to existing fixtures.
3. Measure the real staged-upload/confirmation/initial-chart path. Separate transfer, local preparation, readiness, first chart, background AI and question-to-render latency. Measure isolated backend peak RSS and browser errors; include cold and warm results.
4. Exercise AI off and on, missing/invalid key or provider failure, optional Review data, reload and interrupted work. A model failure must not prevent local charting. Render completed, uncertain and unavailable proposals.
5. Compare deterministic results with independent expected values, verify source preservation, and classify AI outcomes as correct, justified clarification, unnecessary clarification, or confidently wrong. Do not count agreement between models as ground truth.
6. Produce one JSON report and a readable scorecard. Run an initial small trial set as a smoke baseline, clearly labeled; use sufficient repeated observations (prefer at least 100 for release latency evaluation) before making p95 claims. Record hardware, versions, model/settings, sample limits and provider/network conditions without recording credentials or private samples.
7. Fix the largest reproducible correctness/UX defect first, then the dominant measured latency or memory bottleneck. Re-run the affected cases and cross-domain checks; record before/after results.

## Completion criteria

- One documented command produces the baseline report using local fixtures; paid live-AI evaluation is explicitly opt-in and reports unavailable coverage honestly.
- Actual browser interaction covers upload, chart render, AI completion, and optional review; component tests cover failed/empty/uncertain model results. Static type checks alone are insufficient.
- Local aggregates match independent expectations; source values survive adversarial parsing cases; unsupported meaning is clarified.
- The scorecard includes failures and unmet targets as well as successful results, with timing boundaries and reproducible metadata.
- At least one material defect or bottleneck is improved and validated against multiple relevant datasets.

## Known starting evidence

The latest regression showed that a nested provider result (`status`, `decision`) was typed/rendered as a flat decision. It passed static checks and backend tests but crashed React after AI completion. The render regression and live browser repro now pass after correcting the contract. This motivates browser-level acceptance coverage.

The [batching experiment](batched-luna-benchmark-2026-10-02.md) found parallel eight-column batches faster, but with fewer complete decision matches on the existing regression set. Keep batching experimental until accuracy and production integration are validated. Historical taxi ingestion measurements exceed the proposed typical readiness target and exclude browser/transfer time; do not label them as end-to-end readiness measurements.

The [shared deterministic policy evaluation](deterministic-policy-benchmark-2026-10-03.md) removes cardinality-based meaning and guessed units, records abstention/coverage separately, and validates explicit queries on unknown numeric fields. Taxi local readiness remains above target.

## Latest integration: toggle-controlled schema analysis

The [library comparison](library-comparison-benchmark-2026-10-03.md) supports keeping the improved deterministic policy and using GPT-6 Luna for optional semantic suggestions. Upload enrichment now uses one schema request instead of separate Luna column requests; the existing toggle remains the control. Suggestions are role-only, bounded, and advisory. Both storage-engine opt-in tests pass. A live Codex in-app browser check on the real Gapminder demo verified chart readiness before AI completion, optional review, accepting population as a metric without changing parse/unit/aggregation, and successful schema application. Semantic-off still allows the pre-existing independent summary feature; it sends no semantic-analysis request.

Validation: 231 backend tests, frontend helper/render checks and TypeScript checks passed. Browser error logs were empty for the checked session. Native file-chooser automation timed out in the Codex browser; the browser check used the real built-in Gapminder dataset, while the upload endpoints and both storage engines were exercised through integration tests. This is smoke coverage, not a full release acceptance report or an updated latency distribution. The production request uses a 15-second provider timeout; the earlier benchmark used 30 seconds and must not be presented as a new production timing guarantee.

Next product increment: [trustworthy conversational answers](next-milestone-answers.md). The known large-file readiness gap remains open and should remain in the acceptance report.

### Automatic background application

The user chose chart readiness first, then automatic compatible Luna roles. Role-only results now update metadata in the background, preserving existing identifiers, explicit overrides, validated calendar grouping roles, and all source values/parse/unit/aggregation settings. A semantic revision invalidates role-sensitive chart caches while retaining the physical dataset version and already displayed charts. The enrichment polling response carries current column roles to the UI. Tests block the model callback and verify upload returns first, then exercise application on pandas and DuckDB, preservation, stale-job handling and cache invalidation.

An offline replay of the fifteen recorded schema responses through the production application guards yielded 72–73 correct roles, 1–2 unknowns and zero assigned disagreements across 74 columns. This checks the guards against the known failures; it is not a new independent live accuracy estimate. See `backend/benchmarks/results/library-comparison-2026-10-03/automatic-policy-summary.json`.

Live verification of automatic application: in the Codex browser, Gapminder rendered its initial chart while enrichment was pending. Completion updated population from Unknown to Metric, and optional Review data showed the active role with no pending edits. Parsing and aggregation remained Automatic. The displayed chart survived the update, and checked browser error logs were empty. 234 backend tests and frontend helper/TypeScript checks passed. Earlier advisory-flow descriptions above are historical.
