# Overall charts and ingestion priorities — 2026-10-03

## Implemented

Questions asking for an overall sum, ordinary average, median, minimum, maximum or count now return one bar through the existing chart interface. The planner uses `x_axis_key=null` instead of inventing a source grouping column. Overall row counts use an empty measure list; named-column counts count non-missing values. Grouped counts keep their existing semantics. Distinct counts still require clarification.

The engine applies all filters before calculation. Limits/top-N do not truncate the rows entering an overall calculation. All-null numeric sums/averages remain null, and empty filtered datasets retain the existing no-data response. Synthetic chart keys are collision-safe and never used as source-column filters. AI analysis and filter refresh translate chart presentation fields back to the overall execution request; drill-down returns matching source rows through the existing bounded preview. The overall bar displays its value.

Tests caught and fixed an existing DuckDB selection-filter description bug: filtering worked, but a values-only filter was absent from applied-filter text.

## Validation

- Backend: 242 tests passed, including both engines, nulls, negative amounts, empty results, filters, count semantics, invalid columns and synthetic-key collisions.
- Frontend: TypeScript, helper checks and targeted ESLint passed.
- Codex in-app browser: Gapminder overall average, row count, filter refresh, source-row drill-down and AI analysis verified.
- Full public tips (244 rows), diamonds (53,940 rows) and taxi (1,068,755 rows): 12 deterministic engine checks and 9 live GPT-6 Luna planner checks passed (21/21). Expected sums/averages/counts were independently calculated from every source row with csv.DictReader and Decimal. Taxi thousands-separated fares are parsed by the oracle. Numeric comparison uses relative tolerance 1e-9 / absolute 1e-6, not exact accounting equality.

| Dataset | Overall total | Overall average | Rows |
|---|---:|---:|---:|
| tips | 4,827.77 | 19.785943 | 244 |
| diamonds | 212,135,217.00 | 3932.799722 | 53,940 |
| taxi | 21,520,077.81 | 20.135651 | 1,068,755 |

Live overall-query latency: median 1.082s, range 1.021–1.713s across nine calls, excluding ingestion and browser rendering. This is a small targeted feature check, not a broad new accuracy baseline or p95 estimate. No memory improvement is claimed from this run.

Reproduce deterministic full-file checks with `backend/venv/bin/python backend/benchmarks/overall_charts.py`; add `--live` for paid planner checks. Public fixtures must already exist under `/private/tmp/analytico-real-fixtures`; taxi uses the retained full fixture under `backend/datasets`. Detailed live records are `/private/tmp/overall-real-eval-results.json`.

## Proposed speed experiments (not implemented)

The prior eight-dataset baseline remains the source of comparable ingestion measurements. Taxi median local preparation was approximately 13.46s: structure validation 2.17s, CSV loading 3.32s, column validation/type planning 6.15s, typed materialization 1.31s and profiles 0.34s. Phases do not include transfer/rendering. One current taxi import took 14.04s; that single run is not a before/after comparison.

1. **Reduce repeated expensive type checks.** Current full-column projections evaluate many numeric/currency/percent patterns in batches of four columns, with further precision/date checks. First prove simple forms using complete-column inexpensive checks, then run specialized checks only for unresolved columns. Preserve leading-zero IDs, large integers, mixed currency, late malformed values and source/null fidelity. Benchmark on the same eight files and a separate adversarial suite.
2. **Reduce duplicate parsing passes.** Structure validation scans the CSV before strict native loading. Establish parity for ragged rows, quoting, delimiters, encodings, field/row limits and original headers before consolidating checks. The measured 2.17s is an opportunity ceiling for that phase, not a promised saving.
3. **Benchmark resource budgets and scan batching.** Four-column batches protect the fixed DuckDB memory cap. Compare batch sizes/thread budgets on low-memory and ordinary machines; accept changes only with per-process isolated peak memory measurements and preserved wide-file behavior. Do not simply raise memory limits for every installation.
4. **Reuse prepared data on repeat opens.** A persistent local cache keyed by source content plus parser/schema version could avoid preparation of unchanged files. This benefits repeat opens, not first-upload latency, and needs eviction/invalidation design.

Start with option 1. Measure each experiment separately, recording all ingestion stages and isolated peak RSS; keep full-column correctness checks. AI semantic analysis already runs in the background, so making that model faster does not remove the dominant local ingestion cost.

The subsequent [parallelism benchmark](ingestion-parallelism-2026-10-03.md) tested these ideas: four DuckDB threads were the simplest speed/memory compromise on this machine; proof-based pruning helped taxi less, and early AI overlap reduced role coverage. The four-thread configuration has since been promoted with CPU capping and a local override. Other configurations remain benchmark prototypes.
