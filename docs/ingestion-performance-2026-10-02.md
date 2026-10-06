# Ingestion and memory improvements — October 2, 2026

Implemented in order, with three fresh-process trials after each stage. Baseline is commit `d25a254`. All benchmarks disable live providers. Timing starts at CSV parsing and ends after the dataset is stored; fixture generation, transfer and HTTP serialization are excluded. Memory is process-lifetime peak RSS including imports, ingestion and chart execution, not a request allocation or a hard RAM limit. Results are medians on this macOS / Python 3.14 machine with pandas 2.3.3 and DuckDB 1.5.6.

## Measurements

| Stage | 10k rows × 5 columns | 10k × 50 columns | 1M × 5 columns | 1M peak RSS |
|---|---:|---:|---:|---:|
| Before | 0.147 s | 1.104 s | 12.565 s | 745.5 MiB |
| 1. Plain numeric fast path | 0.125 s | 0.709 s | 10.631 s | 749.3 MiB |
| 2. Reuse repeated-value validation | 0.061 s | 0.543 s | 3.771 s | 524.0 MiB |
| 3. Positional array null reinsertion | 0.055 s | 0.447 s | 3.212 s | 507.4 MiB |
| 4. Initial forced disk engine | 0.211 s | 0.764 s | 2.442 s | 271.8 MiB |
| Added precision guards, before optimization | 0.056 s | 0.458 s | 4.351 s | 294.9 MiB |
| Final automatic engine, guarded checks only where needed | 0.073 s | 0.491 s | 2.433 s | 258.4 MiB |

The final automatic engine improves the million-row case by 5.2× and reduces peak RSS by 65.3%. The earlier forced-disk row records its first performance milestone, before extra safety checks. Those checks added cost; applying them only to the relevant decimal/currency/percentage columns recovered the speed without removing validation. It adds startup overhead on smaller inputs. Automatic selection therefore keeps CSV files below 8 MiB on the optimized pandas path and uses disk at or above that threshold. This threshold is a practical heuristic, not a promise about a file's decoded size.

A separate million-row workload with unique numeric values verifies that the improvements are not restricted to repeated amounts: baseline 13.843 s / 909.5 MiB; optimized pandas 4.584 s / 701.6 MiB; initial disk 3.721 s / 289.6 MiB; final automatic disk 3.641 s / 267.7 MiB. These datasets use repeated group/label fields and plain integer measures; unusual formats and very wide/text-heavy inputs can behave differently.

The million-row cold chart also improved from 53.98 ms to 4.37 ms. Warm cached charts remained below one millisecond (0.326 ms before, 0.362 ms after).

Stage 5 moves optional AI off the initial upload path. With a mocked one-second summary request, first response changed from 1.034 s to 0.037 s; enrichment completed in 1.048 s. This demonstrates scheduling behavior, not a live model latency improvement. No provider calls were made and no parsed values or column types changed after enrichment.

Chart response payloads matched the baseline exactly across the staged synthetic trials. Differential tests additionally cover all six supported aggregations, nulls, weighted tail aggregation, collision-free Others labels, date buckets, filters and exact large-integer records. No finite test suite proves safety against every possible input.

Unified-engine results report separate CSV parsing as null when parsing is integrated into disk ingestion; they do not attribute zero time to parsing. Historical unified-engine artifacts had those component fields normalized from their captured phase telemetry; measured total timings are unchanged.

Raw measurements are checked in under `backend/benchmarks/results/ingestion-2026-10-02/`.

The bundled 133 MiB taxi demo (1,068,755 rows × 20 columns) also passed the final browser flow. Its final HTTP handler took 19.625 s, reporting a process peak of 299.4 MiB at ingestion completion; unlike the fresh-worker benchmark, that measurement precedes chart execution. This real, date-heavy dataset remains substantially slower than the five-column synthetic workload. A screenshot and its sanitized handler measurement are included with the raw results.

## Architecture and tradeoffs

The pandas path has a plain-number shortcut before Unicode currency scanning, a capped distinct-value mapper that falls back for high cardinality, and array-based insertion of original null positions. Numeric classification remains row-weighted and validates the entire column. Nullable integer arrays preserve values above JavaScript's exact integer range.

Large inputs use a session-owned disk dataset. It copies source CSV bytes into a private temporary directory, reads bounded pandas chunks with the original CSV null/lexeme rules, and stores raw text plus a typed analytics table in DuckDB. Conversion decisions validate the entire column, including late values across chunk boundaries. Unsupported or ambiguous conversions retain source text. Disk date parsing conservatively preserves timezone-bearing and submicrosecond timestamps as text until timezone/resolution support is implemented. Columns with more than one currency symbol are preserved as text for review rather than combined into one measure. Currency integer values remain exact; large unrepresentable numeric lexemes are preserved. Identifier columns do not become dates merely because their values resemble compact dates. Views, query planning and drilldown request bounded samples; charts execute a closed, parameterized query plan with SQL ordering and limits before fetching rows. Model-generated SQL or Python is never executed.

Disk storage trades local storage and SQL complexity for lower RAM and faster large-data scans. DuckDB has a 64 MB buffer limit, two threads and a 1 GB temporary spill limit. These do not bound total process RSS; measured peak exceeded 256 MiB. Wide validation projections are batched after a forced-wide benchmark exposed an out-of-memory failure. Chart rows remain capped at 500 and preview rows at 1,000. Others collision detection uses a single SQL suffix-gap query instead of a scan per candidate label. Charts must use different columns for the X-axis and measures, preventing conflicting response keys.

Both engines retain the existing session lifecycle: up to ten registered datasets, one-hour idle expiration and eviction cleanup. Short read leases defer closure while a chart, drilldown or sample read is using an evicted handle. Active retired handles temporarily add to the registered dataset count until those reads finish. Replacing a handle retires its old resources. Disk files are temporary session data, not durable saved workspaces; restarting requires re-uploading. Source files supplied by users or demo paths are never deleted. A process crash can leave temporary artifacts, so durable workspaces and crash recovery remain separate future work.

Uploads default to a 256 MiB file cap, 256 columns, five million rows and at most two simultaneous ingestion operations. The row cap is checked before inserting each disk chunk. Multipart transfer happens before the route-level cap and is excluded from these benchmarks; limits here do not replace a transport-level upload limit. Large single fields and decoded string sizes also affect peak RAM.

Optional summary/interpretation work has two workers, bounded pending work and bounded result records. The upload response reports its status; `/enrichment/{dataset_id}` supplies progress/results for the current dataset version. The UI cancels stale polling when data changes. Interpretations are review suggestions and never silently alter parsed values. A setup/provider failure leaves deterministic analytics usable. Up to twelve columns are proposed from small source-grounded samples; summaries use three rows and twenty columns.

## Configuration and reproduction

- `ANALYTICO_INGESTION_ENGINE=auto|pandas|disk` (default `auto`).
- `ANALYTICO_DISK_THRESHOLD_BYTES` (default `8388608`).
- `ANALYTICO_MAX_UPLOAD_BYTES` (default `268435456`).

Use the hash-locked runtime/development requirements; DuckDB is now an exact pinned application dependency. Examples from the repository root:

```sh
backend/venv/bin/python backend/benchmarks/ingestion.py --engine auto --repeats 3 --output /tmp/ingestion-auto.json
backend/venv/bin/python backend/benchmarks/ingestion.py --engine disk --workloads unique --repeats 3 --output /tmp/ingestion-unique.json
backend/venv/bin/python backend/benchmarks/enrichment.py
```

`dataframe` engine reproduces the old direct ingestion benchmark shape with current parsing optimizations. `pandas`, `disk` and `auto` exercise the unified CSV service. The enrichment benchmark uses an explicitly mocked provider delay. Run workloads sequentially on an otherwise idle machine when comparing timings.
