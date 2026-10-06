# Performance improvements — October 2, 2026

Implemented the four performance follow-ups: shared column statistics, bounded deterministic chart caching, bounded interpretation caching with unchanged full-column validation, and accurate parse/ingestion timing plus process-peak memory telemetry. No new runtime dependencies or dataframe engine were added.

## Implementation and limits

- Ingestion computes exact observed distinct counts, first-20 distinct samples, and applicable numeric-year checks once on the cleaned frame. Semantic detection, profiles, default chart selection, and upload summaries reuse this immutable mapping. Null filtering happens on the distinct array, avoiding a full-column temporary copy. Original null, categorical, timezone, ordering, and sample semantics are preserved. This does not skip safety scans on source data.
- Successful deterministic chart responses are cached for five minutes with 256-entry and 24 MiB serialized key/value limits. Keys contain a dataset version and every validated request field, including filters, sorting, limits, grouping, time buckets, and analysis mode. Dataset lookup and axis/semantic validation still run before a hit; the exact previously validated request skips filtering and aggregation. Returned objects are detached copies. Analysis is never cached: every analysis request calls the provider again against computed rows. Different analysis modes have distinct deterministic cache entries.
- Successful or uncertain interpretation decisions are cached for one hour with 512-entry and 8 MiB serialized key/value limits. The key hashes the exact normalized bounded input and includes provider, requested model, prompt version, reasoning effort, and a credential fingerprint. Missing credentials, changed credentials, disabled interpretation, and malformed input cannot bypass their checks through a hit. Provider failures/refusals are not cached. Cached results report `cache_hit=true`, no new token usage, and zero provider latency. Each ingestion still validates the entire column before applying a cached proposal.
- Interpretation caching is explicitly enabled by ingestion. The adapter defaults to `use_cache=False`; live evaluations remain independent and injected transports always bypass the cache. No provider requests were made for this performance validation.
- Both caches are thread-safe, use TTL/LRU eviction, and reject oversized retained entries. Byte limits account for serialized payloads; Python object and cache bookkeeping overhead is additional. Expired entries are removed on access. These are cache retention limits, not upload limits or hard process-RSS budgets. Concurrent cold misses may duplicate work; no new worker or single-flight framework was introduced.
- Every new dataset has a fresh cache version. Existing datasets are immutable in the application's current read paths. Future internal data/schema edits must call `DatasetInfo.invalidate_cached_results()` before serving requests. Expired or replaced datasets never inherit responses from an old dataset instance.

## Measurement scope

Upload and demo handlers start timing before `read_csv_fast`; CSV parsing, raw preservation, cleaning, statistics, classification, profiling, AI summary, metadata, response construction, and storage are measured. Failures produce an error timing record without the filename, data values, credentials, or exception message. Direct dataframe ingestion explicitly labels its timing as excluding CSV parsing. Multipart upload transfer/parsing and FastAPI response serialization are outside these handler measurements.

`process_peak_rss_bytes` is the process-lifetime RSS high-water mark, with a starting high-water mark also recorded. It is shared across concurrent activity, is not current memory or an individual request's allocation, and cannot reset downward after a large upload. It is null on platforms without standard-library resource measurement. Per-workload benchmarks use fresh child processes and include imports, parsing, ingestion, and chart execution in their process peak.

## Validation

Tests cover cache hit/miss behavior, LRU/TTL and serialized-byte bounds, concurrent access, detached values, dataset replacement/expiration/revision, every chart-request key field, fresh analysis calls, failure retry, credential/model changes, explicit transport bypass, and statistics parity for empty/null/nullable/categorical/timezone columns.

A real ingestion regression test supplies two 24-row columns with identical provider inputs. The second contains an unsampled EUR value among USD amounts: it hits the interpretation cache, still requires clarification, preserves the full source column, and generates no automatic metric summary. One mocked provider call serves both ingestions.

Passed 128 backend tests, sixteen evaluation tests, frontend TypeScript, lint, and helper checks. No UI behavior changed, so this batch did not require browser automation.

## Reproducing the benchmark

```sh
backend/venv/bin/python backend/benchmarks/ingestion.py --repeats 3 \
  --output /tmp/analytico-performance.json
```

Workloads are synthetic: 10,000 rows × 5 columns, 10,000 × 50, and 1,000,000 × 5. Each trial uses a fresh process; three repeated chart calls within a trial measure reuse. AI and interpretation are disabled even when a local key exists. CSV generation is outside the measured parse/ingestion time. Results are checked for equality on repeated calls and compared against the baseline for all workloads. The baseline runs the same benchmark script against backend files archived from commit `da8d493` with the same installed interpreter and dependencies.

Recorded measurements and conclusions follow below. These local synthetic timings are evidence for this workload, not production latency guarantees.

## Recorded comparison

Medians of three isolated-process trials per workload. Warm chart time is the median of three repeated calls within each trial. All nine before/after chart results matched exactly.

| Workload | Parse + ingestion, before → after | Repeated chart, before → after | Peak RSS, before → after |
| --- | ---: | ---: | ---: |
| small (10,000 × 5) | 0.143 s → 0.144 s | 1.330 ms → 0.320 ms | 121.5 MiB → 121.0 MiB |
| wide (10,000 × 50) | 1.111 s → 1.098 s | 1.390 ms → 0.321 ms | 150.7 MiB → 149.3 MiB |
| million (1,000,000 × 5) | 12.702 s → 12.474 s | 52.870 ms → 0.339 ms | 699.8 MiB → 745.5 MiB |

Repeated million-row aggregation was approximately 156× faster in this synthetic in-process test. Cold chart execution was still about 53.7 ms; caching does not accelerate the first calculation or measure browser/HTTP rendering. Initial ingestion is largely unchanged: full-column cleaning took about 12.19 s and dominates this workload.

Memory did not improve on the million-row case; measured peak RSS increased by roughly 46 MiB under this benchmark import/execution pattern. Follow-up probes found the high-water mark reached during cleaning, before shared statistics, and sensitive to process/import/allocation history. Removing a full-column temporary from statistics reduced its work but did not reduce this peak. We do not claim a memory improvement or a resolved large-file capacity problem. Upload/process-memory budgets and cleaning allocation optimization remain follow-ups.

Raw synthetic reports: [before](../backend/benchmarks/results/performance-2026-10-02/before.json), [after](../backend/benchmarks/results/performance-2026-10-02/after.json). No real customer data or credentials appear in these reports.
