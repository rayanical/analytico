# Next increment: trustworthy conversational answers

Proposed next increment, October 3, 2026. Direction: [product goal](product-goal.md). This is a plan, not shipped functionality.

## User benefit

A user can ask a business question, get a computed answer with a suitable chart, and refine it without restating everything. Every answer shows enough calculation detail to check it.

## First implementation slice

1. Support overall calculations as a single bar within the existing chart flow (implemented 2026-10-03). Execute through the existing deterministic pandas/DuckDB engine; Luna chooses the plan, never calculates the answer itself.
2. Expose calculation provenance: source columns, aggregation, filters, missing-value treatment, grouping, and time range. Distinguish row count, non-null count and distinct count explicitly.
3. Add reliable follow-ups such as “only last year,” “break that down by category,” and “compare with the previous period.” Bind context to a dataset version, preserve relevant filters, and reject stale context after schema changes.
4. Ask one focused clarification when ambiguity materially affects the result. Do not infer unspecified currency conversions, weights, or business definitions.

## Acceptance

Use held-out real datasets and independently calculated expected answers. Test returns, nulls, numeric codes, ambiguous measures, empty results, filters and schema changes. Score confidently wrong answers separately from abstentions and unnecessary clarification. Measure question-to-render latency against the existing median 3–5 second target; keep local ordinary queries within the existing 500 ms median target. Targets remain targets until measured.

## Performance work that remains

Million-row taxi local preparation was approximately 13.2 seconds in the latest recorded benchmark, above the 8-second median target. Keep profiling its dominant phases and memory while building this answer slice; do not add another ingestion library or an AI readiness dependency. Complete packaging, recovery and broad release acceptance afterward.

## Latest shipped increment and baseline

October 3: display labels share the background Luna schema request, preserve stable keys and source headers, and are editable through optional Review data. Name-only edits retain active roles without re-reading the file. A general ambiguity rule prevents silently choosing a ranking statistic.

The [fresh eight-dataset baseline](fresh-accuracy-2026-10-03.md) records 48 full-file ingestion runs, stage medians, source/null checks and codebook role coverage: local 243/294 correct with 51 unknown, Luna-assisted 284/294 with 10 unknown; neither arm had an incorrect applied role under the rubric. The initial 38/40 query score is retained separately from the post-fix 40/40 regression pass and 18/18 focused follow-ups. At that baseline, overall calculations and reliable conversational context remained unimplemented. Applied filters, aggregation metadata, AI analysis and source-row drill-down already existed; richer calculation evidence needs a specific demonstrated gap before adding UI.

Overall totals, averages, medians, minimums, maximums and counts now use a nullable grouping column and return one bar. Filters, refresh, AI analysis and drill-down reuse the chart flow. See [validation and ingestion priorities](overall-charts-2026-10-03.md).
