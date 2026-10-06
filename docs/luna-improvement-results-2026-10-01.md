# Luna improvement results

The interpreter, chart planner, ingestion summaries, and chart explanations now default to `gpt-6-luna`. The ignored local environment is explicitly set to that model for both model settings. Actual synthetic Responses and Chat Completions calls resolved to `gpt-6-luna`; no fallback model was used. Column interpretation remains opt-in and off by default.

See [official-docs research](luna-improvement-research-2026-10-01.md) for the reasoning behind the prompt and evaluation approach. Strict JSON shape constrains syntax; it does not establish semantic correctness.

## Improvement loop

The original fifteen cases were the development set. A separate Luna agent created thirty holdout cases before seeing candidate scores. Gold labels and category/case IDs were never sent to the provider. The prompt was frozen at `column-interpretation-v3` before scoring the holdout; neither its expected labels nor the prompt were changed to improve holdout scores.

| Candidate | Set | Complete matches | Missed clarifications | Provider failures |
|---|---|---:|---:|---:|
| Original v1, none | Development | 9/15 | 1 | 0 |
| Revised v2, none | Development | 14/15 | 0 | 0 |
| Refined v3, none | Development | 15/15 | 0 | 0 |
| Refined v3, low | Development | 15/15 | 0 | 0 |
| Frozen v3, none, each of three runs | Holdout | 27/30 | 0 | 0 |
| Frozen v3, low | Holdout | 26/30 | 0 | 0 |

The first revision replaced lengthy repeated option descriptions with ordered evidence rules for currency identity, source representation, missingness, date order, stocks, and weighted rates. V2 was too broad about suppressing aggregation when any clarification was required. V3 distinguishes a known requested average from an unknown measurement unit: it can recommend the average while requiring unit clarification. No few-shot fixture answers were embedded in the prompt.

None remains the runtime default. Low did not improve these measured decisions and used 2,822 output tokens for thirty holdout calls versus 1,247 with none. None's pooled latency across ninety holdout requests was p50 1,009.05 ms and p95 1,397.11 ms; low's thirty requests were p50 1,395.1 ms and p95 2,219.56 ms. Runs overlapped in time and provider conditions/caching can differ; this is a small task-specific comparison, not a general model speed claim. Actual billed cost was not returned.

All three none runs agreed on every decision, including their errors. Stable errors are still errors. The remaining differences were:

- Literal `NA` among known-currency amounts: the model correctly requested clarification but chose a concrete parsing policy/aggregation instead of the fixture's unknown parsing policy and no aggregation.
- Unambiguous month-name dates: the model recommended count rather than the contract's none for a temporal key. The scorer flags this automatic decision as wrong; runtime validation rejects it before conversion.
- Numeric codes with unresolved meaning: the model correctly used unknown role/unit and clarification, but chose numeric preservation rather than the fixture's unknown parsing policy.

Low additionally chose `preserve_lexeme` instead of `preserve_source` for numeric category labels. Both preserve values in the runtime, but the exact canonical-label scorer counts the mismatch. Full raw reports are in [the results directory](../backend/evals/results/luna-improvement-2026-10-01/). These seven improvement runs made 165 synthetic interpretation requests; three additional live synthetic calls verified summaries, chart planning, and computed-result explanation on the migrated chat path.

## Deterministic safety fixes

Mocked validated-but-wrong model outputs exercised the real ingestion path. Thirty adversarial proposals cover identifiers, units, locale, date ambiguity, booleans, nulls, non-finite numbers, late unsampled contradictions, and unsupported aggregations. Failures found in the first pass were repaired and the suite rerun:

- Split and common concatenated ID names, and sequence columns, cannot become automatic measures.
- Boolean flags require review before numeric metric treatment.
- A rejected metric proposal cannot fall through to date conversion.
- Parsed numeric values are checked against decimal source values before assignment, including null reinsertion. Values or possible totals beyond JavaScript's exact integer range require review. Higher-precision source values remain untouched when parsing would round them.
- Percent scaling uses decimal arithmetic before conversion to the display/analytics float representation.
- Explicit source currency evidence must agree with the proposed unit, including native numeric columns.
- JSON transports native large integer keys as exact strings. Integer membership and fractional range boundaries avoid float coercion, with an API-level drilldown test for adjacent IDs above 2^53.

An offline replay of the frozen holdout report confirms that uncertain decisions and the remaining incorrect automatic proposal preserve source values and do not generate automatic metric summaries. Valid ISO dates, ordinary currency amounts, percentages, labels, and finite numeric measurements still apply.

Validation passed: 109 backend tests, sixteen evaluation tests, frontend TypeScript, lint and helper checks. The fifteen-case offline baseline predictions remain unchanged with interpretation off. A CPU-only probe of 100,000 rows by five numeric-string columns took 2.463 seconds on this machine, with all five proposed measures accepted. This excludes model/network latency and is a single probe, not a throughput guarantee; full-column safety checks add work for large files.

## Scope of the safety claim

The model can still make a schema-valid semantic mistake, especially about measurement units, entity grain, or the requested calculation. No finite suite proves safety for every possible dataset or adversarial name. The supported claim is that these reviewed cases pass and known unsafe transformations are blocked by application validation, rather than trusted to model judgment.

The classifier still sees bounded samples, unsupported operations still require review, and manual explicit analytics retain their existing rules. Ordinary analytics use floating-point numbers; this is not an exact-decimal accounting engine. A review editor, per-dataset sharing control, broader real-dataset evaluation, persistent workspaces, and asynchronous large-file processing remain product/architecture follow-ups. Re-run the tests and live holdout when changing the prompt or model; recorded offline replays alone cannot detect a future provider-model change.

To reproduce the frozen classifier runs:

```sh
backend/venv/bin/python backend/evals/run_interpretation.py --live --provider luna \
  --cases backend/evals/interpretation_holdout.json --output /tmp/luna-holdout-none.json
backend/venv/bin/python backend/evals/run_interpretation.py --live --provider luna \
  --reasoning-effort low --cases backend/evals/interpretation_holdout.json \
  --output /tmp/luna-holdout-low.json
```
