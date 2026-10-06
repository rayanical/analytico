# Parallel enrichment implementation and follow-up measurements

Production enrichment now overlaps the business summary and column interpretation through a shared pool of four requests. It considers all supported columns rather than the first twelve, publishes completed proposals during processing, and exposes coverage in the upload panel. Deterministic ingestion still completes before optional AI results; proposals never change the active schema automatically.

## Implementation and tradeoffs

`services/parallel_enrichment.py` owns the request scheduling. At most four callbacks execute across the two active dataset jobs, with at most four outstanding requests per job. Each request retains the provider adapter's timeout. A sixty-second job budget stops scheduling, cancels queued work, and preserves completed results. In-flight requests cannot be forcibly interrupted, but their late results cannot publish after the worker finishes. Dataset-version checks run before scheduling, before a queued callback invokes the provider, and before processing results. The manager rejects stale-version and terminal partial updates.

Provider outage or HTTP configuration failures stop additional column requests, while already-running work can finish. Individual invalid decisions or refusals remain per-column failures. Exceptions expose generic messages only. Status coverage includes total, selected, completed, failed and skipped column counts, plus a stop reason. `complete` means every column request finished without a provider failure; it does **not** mean every decision was actionable or semantically correct. An uncertain decision still requires review.

Full coverage is the default within the budget and the existing 256-column ingestion limit. Wide files can issue more requests than before and consume more tokens; they can also hit the budget. Those limits and failures are now visible rather than silently hidden behind a completed job. Interpreter-off configurations can still generate a summary; their column coverage remains incomplete.

## Live before and after

Two isolated-process runs per dataset and strategy. GPT-6 Luna, reasoning none, existing prompt, original bounded source samples, application interpretation cache bypassed. Comparisons use the previous saved sequential runs. Filesystem caching, provider caching and network conditions are uncontrolled; two repetitions are insufficient for percentile or general latency guarantees.

| Dataset | Previous sequential AI, after data ready | New production parallel AI | Coverage before → after |
| --- | ---: | ---: | --- |
| Taxi: 1,068,755 rows × 20 columns | 14.90 s | 7.15 s | 6–9 → all 20 |
| Gapminder: 1,704 rows × 6 columns | 8.43 s | 2.04 s | all 6 → all 6 |
| Wide: 10,000 rows × 50 columns | 13.94 s | 14.20 s | 9–11 → all 50 |

Taxi background time fell approximately 52% while covering more columns; Gapminder fell approximately 76% with equal coverage. Wide-file elapsed time stayed roughly the same while completing every column. This is not an equal-work speed comparison for taxi or wide: the old worker stopped early.

New production data-ready medians were 20.92 seconds for taxi, 0.050 seconds for Gapminder and 0.625 seconds for wide. No ingestion algorithm changed, so differences from earlier ingestion timings should not be attributed to the enrichment change. Production background ranges were 5.92–8.37, 1.97–2.11 and 13.12–15.28 seconds respectively. Initial column roles matched the previous runs exactly, and initial profiles matched within 1e-9 floating-point tolerance. Each worker asserted the schema remained unchanged through enrichment.

The twelve new timing runs issued 196 column calls and twelve summaries. Column outcomes: 48 actionable, 148 uncertain, zero provider failures. Classification tokens were 208,662 input and 7,809 output; the standard-rate estimate is $0.02477, excluding summaries whose helper does not expose token usage. This is not observed billing; see the [previous benchmark's pricing source](dataset-ai-latency-2026-10-02.md).

## Upload to first natural-language chart

A follow-up measured the actual local staging → confirmation ingestion → immediate `run_query` path with production background enrichment enabled. Two repeats per dataset in one process, GPT-6 Luna, provider/filesystem caches and network uncontrolled. All four requests returned nonempty charts. Questions were total taxi tips by payment type and average Gapminder life expectancy by continent.

| Dataset | Staging and preview | Confirmed ingestion | Question to chart response | Total processing |
| --- | ---: | ---: | ---: | ---: |
| Taxi, 133 MiB | 2.48–3.60 s | 18.67–19.00 s | 1.10–1.87 s | 23.35–23.37 s |
| Gapminder, 80 KiB | 0.051–0.059 s | 0.047–0.048 s | 1.19–1.66 s | 1.29–1.76 s |

These totals exclude browser file transfer/multipart parsing, user confirmation/thinking/typing/speaking and browser rendering. They are backend processing observations for two simple grouped-chart questions, not full browser wall-time or a guarantee for arbitrary questions. Chat becomes enabled as soon as the dataset is ready; it does not depend on completion of background enrichment. Results: `backend/benchmarks/results/parallel-ai-2026-10-02/upload-to-chart.json`.

## Cautious hybrid experiment

The benchmark-only `hybrid_cautious` strategy routes numeric columns when distinct counts are small, values are nearly unique, leading-zero lexemes appear, or source samples are not finite plain decimals. It also routes categorical, identifier and unknown roles. Thresholds are heuristics, not calibrated semantic confidence; no dataset names or keyword regex were added. The production worker uses all columns and does not call this gate.

| Dataset | Selected columns | Background time |
| --- | ---: | ---: |
| Taxi | 15/20 | 4.89 s |
| Gapminder | 5/6 | 2.02 s |
| Wide | 2/50 | 1.21 s |

On the existing thirty-case offline role regression, the cautious gate increased correct labels from the earlier hybrid's 27/30 to **28/30**, but increased model calls from 15 to **27**. All-AI remained 29/30. This reuses saved predictions and previously used fixtures; it is not new live accuracy evidence. The all-null field remains locally misclassified, and a selected ordinal risk band retains the model's incorrect identifier role.

Eighteen separately generated routing stress cases further demonstrate the limitation. Six formatting/low-cardinality/near-unique cases routed successfully. Twelve integer-label cases with 33, 128 or 1,001 distinct district, product-variant, taxonomy or account codes were skipped when supplied a local metric role. Their business meaning is known to the test, but cardinality does not establish it. These are routing counterexamples, not live AI accuracy tests; they explain why the hybrid stays experimental rather than claiming general safety from the 93.3% replay score.

A safe selective default needs stronger evidence, such as a confirmed schema or data dictionary, or a cheap semantic decision for every column before skipping further interpretation. A local metric role and numeric validation alone cannot prove additive meaning. Batching independent semantic decisions is a possible next experiment, but has not been measured here.

## Taxi ingestion profile

One provider-disabled cProfile run measured approximately 24 seconds inside ingestion, versus approximately 21–22 seconds without profiling. Cumulative timings overlap and include profiler overhead:

- CSV chunk ingestion: 8.43 seconds, including 86 append operations totaling 6.97 seconds.
- Analysis and parsed-view construction: 8.24 seconds.
- Full structural CSV validation: 4.32 seconds.
- Source preservation: 2.52 seconds in this run, including filesystem/cache effects.

The cap of 250,000 cells per CSV chunk creates approximately 12,500-row batches for this twenty-column file. Repeated inserts and conversion/statistics scans deserve the next controlled ingestion experiment. Raising the chunk cap trades memory for fewer inserts; bulk loading must preserve current source lexemes, CSV settings and late-row validation. This task profiles those opportunities without removing checks or claiming an unmeasured ingestion optimization. The safe summary of the bottleneck is local scans and database work, not Luna latency.

Reproduce the profile from the repo root:

```sh
OPENAI_API_KEY='' COLUMN_INTERPRETER=off OPENAI_MODEL=gpt-6-luna \
  backend/venv/bin/python -m cProfile -o /private/tmp/taxi-ingestion.pstats \
  backend/benchmarks/dataset_ai_latency.py \
  --worker backend/datasets/2021_Green_Taxi_Trip_Data_20260221.csv \
  --label taxi --strategy off --output /private/tmp/taxi-profile.json
```

## Validation and artifacts

205 backend tests and sixteen evaluation tests passed offline. New regression checks exercise actual ingestion beyond twelve columns, summary/column overlap, partial success alongside failure, four-request concurrency shared across jobs, time-budget retention, stale jobs and updates, provider outage handling, and extended disk payload equivalence. Frontend lint and TypeScript passed; the production webpack build passed. Turbopack failed to bind its build worker port in this environment, so validation used Next's supported webpack build with access to the existing Google Fonts assets.

Measured artifacts: `backend/benchmarks/results/parallel-ai-2026-10-02/` contains all live trials, their summary, cautious replay scores, fresh routing stress cases and the safe profiler summary. The first two production taxi trials do not include the new optional coverage field in the benchmark JSON; the production endpoint and tests do verify coverage.

Reproduce live measurements with a configured key (incurs usage):

```sh
COLUMN_INTERPRETER_MODEL=gpt-6-luna OPENAI_MODEL=gpt-6-luna \
  backend/venv/bin/python backend/benchmarks/dataset_ai_latency.py \
  --live --strategies current hybrid_cautious \
  --output /private/tmp/parallel-ai-results.json
```
