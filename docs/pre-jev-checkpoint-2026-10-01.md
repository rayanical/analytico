# Pre-Jev checkpoint — October 1, 2026

Completed the requested baseline cleanup, contract reconciliation, dependency remediation, and interpretation benchmark. Jev integration and CI were not part of this batch.

Subsequent dependency-locking follow-up: backend runtime and development install requirements now contain exact versions and hashes, with editable ranges in `.in` files. See [dependency maintenance](dependency-locking.md). The frontend continues to use `package-lock.json` and `npm ci`.

## Cleanup and contracts

Current architecture and README now describe the removed execution path, optional AI data sharing, keyless manual workflows, included versus optional demos, parsed drilldown rows, and temporary storage accurately. Original audits and the numeric-parsing proposal are marked historical rather than presented as current behavior.

Upload metadata uses matching backend/frontend enum values, including general-format and unknown-role fallbacks. Default-chart metadata excludes the response-only empty state. Query sorting passes through the frontend request. Exact nullable chart/filter wire types are normalized at the HTTP adapter for UI consumers; this is typing and normalization, not a general runtime response validator. Drift regressions cover explicit null fields, persisted metadata, and response-authoritative time buckets.

Column interpretation now uses the shared configured OpenAI model rather than a hardcoded schema-enrichment model. This does not migrate the provider/API or promise compatibility with every model name.

Dependency testing exposed the Pandas/PyArrow read path stripping leading zeros before dtype=str conversion. CSV ingestion now uses the C parser with explicit string typing and blank-only missing values. PyArrow was unused elsewhere and removed from declared dependencies. This keeps lexemes consistent at the cost of dropping the alternate parsing path; larger-file performance still needs measurement.

## Verification

- 53 backend tests passed with OPENAI_API_KEY disabled.
- Six benchmark tests and 15-case schema validation passed. Offline snapshot regeneration matches the checked-in baseline.
- Frontend ESLint, helper tests, production build, and TypeScript passed.
- git diff --check passed.
- Updated production app loaded Gapminder, pinned a dashboard chart, generated a PDF using jsPDF 4.2.1, and restored dataset metadata/history on reload. Browser console had no errors; transient Recharts sizing warnings remain.
- PDF structure and rendering confirmed the export works after the dependency upgrade. Existing report-layout issues remain: widget controls/scrollbars appear and the bottom axis label is clipped. This check is compatibility evidence, not proof of general report fidelity.
- No live model calls were made. Dependency audit results and version changes are in the separate dependency report.

## Interpretation baseline

The 15 curated synthetic cases cover IDs, currency identity, rates and denominators, balances, date/number ambiguity, unknown units, mixed text and missing observations. Expected decisions remain separate from exported candidate inputs to prevent answer leakage. The snapshot calls the current runtime inference; it never reads expected decisions.

Current runtime column-role accuracy is 7/15 (46.7%). Aggregation accuracy is 2/6 emitted decisions (33.3%), with 40% coverage. Units, explicit parsing policy and clarification are not emitted by the runtime, so their accuracy is unscored rather than zero or perfect. These are diagnostic results on a small deliberately challenging fixture, not estimated accuracy on customer data.

The benchmark includes concepts such as ratio-of-sums and snapshot aggregation to describe correct interpretations; those operations are not implemented by the chart engine. A future interpreter must still respect the executor's supported capabilities.

## Next change

Evaluate one bounded Jev column-interpretation adapter against this baseline and retain deterministic validation/parsing/execution. Durable local workspaces, installation, resource budgets, privacy switches, runtime contract generation/validation, larger-file testing, and report/accessibility polish remain separate work.
