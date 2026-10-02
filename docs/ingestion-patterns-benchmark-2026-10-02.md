# Consolidation, ingestion patterns and Luna comparison

**Production follow-up:** the native loader is now the default for disk-backed
ingestion, with the compatibility fallbacks described below. The original
comparison numbers remain experimental measurements. After promotion, all 212
backend tests passed. A default-path taxi staging/import/GPT-6 Luna chart smoke
test, without loader patches, passed: 2.191 s staging, 14.185 s ingestion and
3.132 s question processing, 19.508 s total. This is one run, with provider latency
variation, excluding transfer, human input and rendering. The native loader was
verified active and all 1,068,755 rows were retained. Results are in
`backend/benchmarks/results/native-production-2026-10-02/`.

The production UI consolidation is committed as `689153f`: valid uploads prepare
automatically, and one optional Review data action contains parsing, cleaning
details and column edits. Failed imports retain their source for optional recovery.
See [the consolidation checks](consolidated-review-2026-10-02.md).

## Research conclusion

[The primary-source research](ingestion-patterns-research-2026-10-02.md) covers
DuckDB, Polars, Power Query and Tableau. Relevant mechanisms are native bulk
loading, bounded preview/inference, lazy or vectorized analysis, and materialized
analytics storage. These tools separate fast previews from complete processing;
sampled type inference is not a guarantee about every row.

We tested the native DuckDB loading mechanism against our existing implementation.
We did not run Power BI, Tableau or Polars, so the numbers below are not product
speed comparisons. No new runtime dependency was added.

## Equal-work results

Two fresh-process trials per arm on this machine. Medians, with AI disabled for
the loader comparison. Both paths retain original source bytes, validate every
record, store raw text, run the same full-column conversion/profile logic and
execute the same deterministic grouped chart.

| Dataset | Current data ready | Native data ready | Reduction | Current → native peak RSS |
|---|---:|---:|---:|---:|
| Taxi: 1,068,755 rows × 20 columns | 20.064 s | 14.313 s | 28.7% | 316.3 → 282.0 MiB |
| Synthetic: 1,000,000 rows × 5 columns | 2.927 s | 2.230 s | 23.8% | 263.0 → 220.7 MiB |
| Wide: 10,000 rows × 50 columns | 0.687 s | 0.593 s | 13.7% | 191.3 → 184.7 MiB |
| Gapminder: 1,704 rows × 6 columns | 0.051 s | 0.051 s | unchanged | 134.7 → 134.6 MiB |

Taxi loader-only time fell from 8.197 s to 3.284 s; complete processing still
takes substantially longer than loading. The taxi native preparation range was
13.175–15.451 s; current was 19.359–20.768 s. These two sequential trials are
directional evidence, not statistical confidence intervals. OS cache and machine
load were uncontrolled. Wide was forced through disk storage to exercise the
loader; the normal 8 MiB threshold and small-file pandas path remain unchanged.

Preparation here excludes import staging, browser transfer, natural-language
planning and rendering. Peak RSS is the isolated process lifetime high-water mark,
including imports and chart execution; it is not the DuckDB buffer limit or
retained dataset RAM. Temporary disk consumption was not measured in this round.

Schema and profile payloads and chart results passed parity checks. Integers and
strings compare exactly; floating-point aggregates use relative tolerance 1e-10
and absolute tolerance 1e-8, because parallel sums varied even between repetitions
of the original loader. This tolerance does not permit identifier conversion or
large-integer precision loss.

## Luna in the mix

Live trials used **GPT-6 Luna** for existing background summary/column proposals.
They sampled the same bounded values/context and did not change parsed schema.
This tests the current advisory design alongside the faster loader; it does not
test an LLM parsing every row or automatically choosing a new schema.

| Dataset | Native data ready with Luna | Remaining background AI | Proposal status in each trial |
|---|---:|---:|---|
| Taxi | 13.926 s | 11.637 s | 9 ok, 11 uncertain |
| Gapminder | 0.053 s | 4.326 s | 2 ok, 4 uncertain |
| Wide | 0.576 s | 18.977 s | 2 ok, 48 uncertain |

All selected columns completed, with no failed or skipped requests. Completion
does not mean every proposal is suitable for applying. The uncertainty statuses
come from the current model/policy guard contract, not a labeled accuracy score.
For example, taxi proposals identified trip distance and fare/total amounts as
metrics, but those proposals remained uncertain; the deterministic schema currently
classifies those fields as identifiers. This exposes a semantic limitation worth
testing next, not permission to silently apply sampled decisions.

Charts execute before enrichment completes. Adding Luna does not accelerate raw
CSV loading; the benefit to evaluate is meaning, units, codes and useful chart
defaults. The wide-file result argues against blocking every upload on one request
per column. A batched bounded schema proposal is a reasonable next experiment,
with labeled heterogeneous fixtures, independently checked conversions and
explicit ambiguity handling. Previous numeric-only routing experiments missed
numeric category codes, so those heuristics remain disabled.

## Complete preparation-to-natural-language-chart check

Two additional fresh-process trials per arm measured actual staging, automatic
confirmation and an immediate GPT-6 Luna natural-language chart request, while
production background enrichment was active. Both the requested axes and aggregation
were independently checked; chart data also passed cross-arm parity.

| Dataset / path | Staging | Ingestion | Question → chart | Total processing |
|---|---:|---:|---:|---:|
| Taxi / current | 2.511 s | 18.868 s | 2.400 s | 23.778 s |
| Taxi / native | 1.798 s | 13.319 s | 1.479 s | 16.596 s |
| Gapminder / current | 0.043 s | 0.048 s | 1.714 s | 1.805 s |
| Gapminder / native | 0.043 s | 0.048 s | 1.702 s | 1.793 s |

Taxi total processing ranged 23.358–24.199 s currently and 16.588–16.603 s natively.
The loader does not explain the entire total difference: staging and provider
response times also varied. The controlled offline ingestion comparison above
isolates the more repeatable local benefit. These are backend processing timings,
excluding file transfer, human speaking/typing/thinking, HTTP serialization and
browser rendering. The natural-language questions were total tips by payment type
and mean life expectancy by continent. This is not a claim about all possible chart
questions or about a 3-second complete taxi workflow.

## Compatibility and verification

The experiment only patches the raw loader inside benchmark processes; production
continues using the established reader. Native loading preserves VARCHAR source
lexemes and parameterizes source/options, with generated positional field names.
It uses strict parsing, no skipped rejects, two threads and the existing 64 MB
DuckDB buffer limit. An 8 MiB CSV buffer avoids the enormous allocations produced
by raising the maximum native line size without explicitly sizing its buffer.

Tests found concrete incompatibilities: native single-column blank-line behavior
differs from pandas, and an empty null-token list is rejected by DuckDB. The
prototype retains pandas for single-column and non-UTF-8 files, represents the
empty null policy with force-not-null, and falls back after discarding partial
native storage when bounded native parsing rejects a supported long/wide record.
The long-record fallback is tested against expected complete source values.
These are parser capability fallbacks, not dataset-name special cases.

Five new promotion-gate tests cover locale/custom null rules, empty strings,
quoted multiline fields, literal NA, exact large integers, CP1252/UTF-16 fallbacks,
150,000-row native ordering, late malformed records, blank lines and long-record
fallback. All 17 existing disk tests also pass with the experimental loader patched
in, including its fallback paths. The full 212-test backend suite passes. Frontend
lint/type/helper checks and the production webpack build pass. Browser/API coverage
and the file-chooser limitation are recorded in the consolidation note.

Raw results, summaries and the optional-review screenshot are under
`backend/benchmarks/results/patterns-2026-10-02/`. Reproduce from the repo root:

```sh
backend/venv/bin/python backend/benchmarks/ingestion_patterns.py --repeats 2 --output /tmp/patterns.json
# Explicit live calls require the configured OpenAI key:
backend/venv/bin/python backend/benchmarks/ingestion_patterns.py --live --repeats 2 --output /tmp/patterns-live.json
backend/venv/bin/python backend/benchmarks/ingestion_patterns.py --live --flow --repeats 2 --output /tmp/patterns-flow.json
```

The next production change should promote the native loader with these compatibility
guards and broader real-file coverage. Keep full validation, then optimize remaining
scans using phase measurements. Investigate a smaller batched Luna proposal separately;
do not remove parser validation or apply uncertain transformations to chase latency.
