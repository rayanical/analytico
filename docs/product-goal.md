# Analytico product goal

Status: agreed vision and initial acceptance targets, October 2, 2026. Targets describe the intended product, not current measured capabilities. Change this document deliberately when the product direction changes. Historical audits and benchmark reports remain evidence; they do not supersede this goal.

## Product promise

A local analyst that turns an unfamiliar CSV into trustworthy, useful answers within seconds, and makes those answers easy to inspect and present.

Differentiation: fast, trustworthy analysis of messy files on the user's own computer. Users bring their own AI key. Manual exploration works without AI. Optional AI sends bounded dataset context to the configured provider; local operation does not imply that AI requests remain on-device.

Working audience assumption: business analysts and operators exploring exported CSVs. Working version-one capacity assumption: make 1M rows × 50 columns excellent before expanding toward 5–10M rows. These two assumptions still need explicit product confirmation. The agreed large-file timing workload below is 1M rows × 20 columns; do not silently extend its target to wider or larger files.

## Intended experience

1. Drop a CSV into the app, optionally enabling AI column semantic analysis before upload.
2. Explore immediately after local preparation. Optional AI analysis runs in the background.
3. Ask a question and get a direct answer, a useful chart where appropriate, and an inspectable calculation explanation.
4. Refine the result conversationally, including filters, breakdowns and comparisons.
5. Pin findings into a polished dashboard and export them.

Review data is optional. Safe automatic changes must satisfy explicit local rules and full-column checks. Uncertain values retain their source representation. Business ambiguity that materially changes the answer warrants a focused clarification. The AI column semantic analysis toggle enables one bounded GPT-6 Luna schema request in the background. It suggests roles for unknown or weakly supported columns using table context and existing full-column statistics. Parsed values, units and aggregations stay unchanged. Compatible strong role interpretations apply automatically when background analysis completes; existing identifiers, explicit user choices and calendar grouping conflicts remain protected. Original values, parsing, units and aggregations are unchanged. Review data is for optional corrections and unresolved cases. Background role changes invalidate semantic chart caches and update UI column metadata without replacing an already displayed chart. Model self-confidence is not proof of business meaning. Schemas over 64 columns or 64 KiB of context currently retain local detection without a schema request.

## Performance acceptance targets

Measure on a named reference laptop, using fixed fixtures with recorded file sizes and data characteristics. Record distributions and failures rather than reporting only the fastest run.

| Experience | Initial target |
| --- | --- |
| Upload feedback | Visible progress within 200 ms |
| Small CSV, up to 100k rows × 20 columns | Ready to explore within 2 s |
| Large reference CSV, 1M rows × 20 columns | Typical ready time within 8 s; p95 within 12 s |
| Ordinary filtering and aggregation | Typical within 500 ms; p95 within 1 s |
| Natural-language question to rendered answer | Typical 3–5 s; p95 within 10 s |
| Optional AI column analysis | Background work never gates readiness or charting |
| Backend memory, 1M-row reference workload | Target peak process RSS below 500 MB; avoid repeated full-data copies |

Typical means median for benchmark reporting. Backend process RSS and browser memory are measured separately. Readiness means the browser can inspect the prepared dataset and request charts, not merely that parsing completed. Question latency ends when the browser renders the answer/chart. Upload transfer, staging, validation, profiling, model calls and rendering must be reported separately as well as end to end. Wide files, long text, complex queries and larger workloads need separate tested limits. Provider/network conditions must accompany AI measurements. Small trial sets cannot establish p95 reliably.

## Trust and reliability contract

- Calculations use the deterministic analytics engine; models interpret intent and meaning, never execute generated Python.
- Answers expose the selected measure, aggregation, filters and relevant assumptions.
- Original values remain recoverable. Ambiguous conversions never silently destroy information.
- Missing observations remain missing unless the user explicitly chooses an imputation policy.
- Ambiguous business questions receive useful clarification rather than a confident guess.
- Malformed files, model failures, interruption and expired dataset state leave the app usable.
- Evaluate answer correctness, confident wrong answers and unnecessary clarification across diverse domains. Successful HTTP requests and valid output schemas do not establish correctness.
- No known silent data loss, incorrect deterministic aggregate, or React crash in the acceptance suite is acceptable for release. Establish a diverse baseline before choosing broader model accuracy thresholds.

## Incremental roadmap

1. **Dependable upload-to-answer flow:** establish one repeatable cross-dataset acceptance report; repair the largest demonstrated failures and bottlenecks. See [first milestone](first-milestone.md).
2. **Trustworthy dataset understanding:** separate parsing safety from semantic uncertainty; improve bounded Luna analysis and automatic application only against the acceptance suite.
3. **Useful conversational analysis:** direct answers and calculation provenance, reliable follow-up context, focused clarification and predictable supported operations.
4. **Presentation and release quality:** cohesive dashboards/export, installation, local recovery, resource budgets and actionable errors.

Each increment needs a user-visible benefit, relevant before/after evidence, correctness checks, and documented limits. General improvements should pass unrelated datasets and adversarial cases. Avoid dataset-specific mappings, speculative rewrites, and additional mandatory review steps.
