# Column libraries versus targeted Luna: measured comparison

Measured 2026-10-03 on this local arm64 macOS machine. This evaluates the [library shortlist](column-inference-libraries-2026-10-03.md), with actual Visions and DataProfiler installations and actual GPT-6 Luna requests. It does not change production inference, source values, or application dependencies.

## Decision

Keep the current source-preserving pandas/DuckDB pipeline. On this workload, neither library's native semantic outputs resolved any of its remaining unknown roles when replayed as an unknown-only enrichment layer. Visions is a fast logical-type detector; DataProfiler's actual pretrained labeler mostly recognized numeric storage/lexical types rather than our analytical meaning. Neither warrants replacing the current policy for this purpose.

Luna is the useful semantic challenger: it recovered substantially more roles, but its qualitative “strong” assessment still allowed disagreements with known key and calendar usage. Keep enrichment advisory until its promotion rules handle those conflicts. This experiment does not establish universal semantic accuracy.

## Workload and protocol

Five complete real sources: UCI hourly Bike Sharing (17,379 × 17), UCI bank-full (45,211 × 17), UCI Online Retail (541,909 × 8, spreadsheet exported to CSV), repository NYC Green Taxi (1,068,755 × 20), and UCI white Wine Quality (4,898 × 12). Source/codebook labels and accepted alternatives are in `backend/evals/real_dataset_manifest.json`. They were not supplied to either library or Luna. No synthetic performance files were created.

`library_inputs.py` validated local ingestion and row counts, then selected 1,000 evenly spaced actual records including the first and last from each complete CSV. It recorded original lexemes/nulls, complete-column statistics, source hashes and current usage metadata. Library detectors received only the bounded values; they ignored local usage and gold labels. Luna received names, all-column context, twelve spread values per column and complete statistics; it received no filename, source hash, gold labels or existing roles. These are intended deployment configurations, not an assertion of identical information or interchangeable tasks.

Each library ran in three fresh subprocesses. Luna ran three requests per dataset, with schema order shuffled in repeats two and three: **15 successful live requests**, all explicitly using `gpt-6-luna`, reasoning `none`, strict structured output, and no result cache. One request covered a dataset's selected columns. The selection rule treated unknowns and all evidence except explicit user roles, validated datetimes and validated calendar years as weak. It selected **70 of 74 columns**, not just the fifteen unknowns; this is an evidence-based selector, not a calibrated confidence score.

For Luna, role proposals were merged into a separate experimental dictionary only when its self-reported strength was strong and the role fit physical/profile constraints. Temporal component proposals required compatible full-column ranges; metrics required nonempty numeric storage. Category/identifier checks do not prove business meaning. Units, source conversions and aggregation decisions were intentionally independent and unchanged. No proposal was applied to the running app.

The “original” row below is the captured immediate pre-policy deterministic baseline from the same four files, not the earliest repository version and not a fresh replay of obsolete code. The current snapshot was regenerated during input preparation and matched the previous final policy. Wine has no captured original measurement, so the paired comparison uses four datasets / 62 columns.

## Paired role results

| Arm | Correct / 62 | Unknown | Assigned-role disagreements |
|---|---:|---:|---:|
| Original recorded deterministic policy | 53 (85.5%) | 0 | 9 |
| Current local policy | 48 (77.4%) | 14 | 0 |
| Visions: native logical-type evidence | 10 (16.1%) | 52 | 0 |
| DataProfiler: native semantic labels | 4 (6.5%) | 58 | 0 |
| Current policy + Luna, trial 1 | 57 (91.9%) | 2 | 3 |
| Current policy + Luna, trials 2 and 3 | 58 (93.5%) | 1 | 3 |

**Interpret the library rows carefully:** the scoring target is identifier/metric/temporal/categorical business usage. An `Integer` or `Float` result is left unknown for that task. The low role coverage does not mean that these libraries failed to recognize numbers. Visions inferred a logical type for all 74 columns, and DataProfiler's actual neural labeler ran on 73 nonempty columns per repetition; the all-null taxi field remained unknown without a model call.

The frozen library mapping treated temporal classes as temporal, categorical/Boolean classes or explicit ordinal labels as categorical, and supported entity-key labels as identifiers. DataProfiler `QUANTITY` could map to metric, but no native quantity label resolved the unknown count columns. A separate diagnostic treating pure numeric outputs as metrics produced **34/62 with 17 disagreements for Visions**, and **28–29/62 with 19 disagreements for DataProfiler**. These are convenience proxies added by the adapter, not native business-meaning guarantees. They recreate the problem of promoting codes and IDs to measures.

A separate integration replay used each library's native semantic role only where the current role was unknown. Both stayed **48/62 with fourteen unknowns**, and **59/74 with fifteen unknowns** across all five datasets. This is a replay of recorded proposals, not a separately timed production integration.

Across all five datasets / 74 columns, the current policy scored **59 correct, fifteen unknown, zero assigned disagreements**. The Luna merge scored **69–70 correct (93.2–94.6%), one or two unknown, three disagreements**. Wine added twelve correct Luna roles in every trial; current local inference had eleven correct and an unknown quality score. Visions' primary role mapping had ten correct / 64 unknown. DataProfiler's native mapping had four correct / 68 unknown / two disagreements: wine `fixed_acidity` and `residual_sugar` were given category roles by the mapped entity labels. The role score does not evaluate the correctness of every finer-grained entity label.

### What Luna still got wrong

In all three trials, `PULocationID` and `DOLocationID` were proposed as categories rather than the codebook's identifiers. Bike `weekday` was proposed as temporal rather than its categorical grouping role. These distinctions can overlap conceptually; the benchmark keeps its original contract rather than changing gold labels after seeing results. They are role-contract disagreements, not evidence of an arithmetic error in a generated chart.

The first trial kept bike `yr` unknown after rejecting a strong temporal/year proposal: its values are 0/1 codes, not calendar years. Later trials correctly suggested categorical. Bike `instant` was resolved as an identifier in every trial. The all-null taxi `ehail_fee` stayed unknown. An entirely empty column cannot acquire reliable meaning from values alone.

These cases show why nonempty/numeric validation and model self-confidence cannot prove semantics. The next promotion policy should preserve user choices, require review for changes to established key usage, and represent calendar components explicitly rather than silently replacing a grouping role. A stricter policy needs a new held-out evaluation; it is not already validated by this run.

## Performance and memory

These are **enrichment proposal costs**, excluding original upload/transfer, full ingestion, browser rendering and package installation. Library timings process 1,000-value samples per column; they are not full-file scan timings. Each worker handles all five datasets in sequence after one library/model initialization.

| Candidate | Setup/import in a fresh worker, median | Warm proposal work across all five | Fresh process wall across all five |
|---|---:|---:|---:|
| Visions 0.8.2 | 0.309s import + typeset | 0.485s inference | 1.020s |
| DataProfiler 0.14.0 + pretrained ML | 2.849s imports + 0.285s model construction/load | 3.349s profiling + labeling | 6.916s |
| GPT-6 Luna | Existing app client; network calls measured | 2.27–4.06s median per schema | No local model initialization; fifteen requests measured separately |

Visions' isolated worker median peak RSS was **110.3 MB**. DataProfiler's median resident-memory snapshot after all five profiles was approximately **783.2 MB**; a separate single-run high-water probe measured **784.7 MB**. The snapshot and single probe are different measurements. Luna's local client process peak was **90.5 MB**, excluding OpenAI's remote model memory. These cannot be added directly to the app's measured RSS or interpreted as total packaged-app memory. The current ingest worker's earlier peak included full ingestion and reference calculations; the library workers do not.

An initial standalone DataProfiler probe recorded 21.55s under a different timing boundary that included imports. It is not comparable to the formal 0.285s post-import constructor measurement and does not prove a cache speedup. The 6.916s complete fresh-worker wall measurement includes imports, model initialization and all five profiles. Three trials are not a P95 estimate; concurrent work on this shared machine also affects timing.

Per-schema warm proposal costs:

| Dataset | Visions | DataProfiler profile + labeling | Luna |
|---|---:|---:|---:|
| Bike Sharing | 0.102s | 0.790s | 3.487s |
| Bank marketing | 0.079s | 0.742s | 3.816s |
| Online Retail | 0.049s | 0.450s | 2.269s |
| Green Taxi | 0.192s | 0.830s | 4.056s |
| White Wine Quality | 0.065s | 0.535s | 2.879s |

Per-dataset medians do not necessarily sum to the median whole-worker duration. The small library times are possible because neither library profiles a million taxi rows: it proposes from the bounded sample. Luna's taxi request also uses bounded evidence; its network latency is not proportional to all 1,068,755 rows.

For context, the previous three-trial full local-ingestion taxi median was **13.155s**, compared with the recorded original **13.378s**. If the user must wait for this Luna enrichment as well, adding the independently measured medians suggests approximately **17.2s**, before upload transfer or browser rendering. That sum is an estimate, not a newly measured end-to-end request. Background enrichment can leave initial local readiness near the original ingestion time. This experiment does not measure the subsequent natural-language-to-chart request.

## Model usage and cost

All fifteen live requests reported **33,300 input tokens and 4,485 output tokens**, with zero cached input tokens. Using the standard prices listed in [official OpenAI documentation for GPT-6 Luna](https://developers.openai.com/api/docs/models/gpt-6-luna), $0.10 per million input tokens and $0.50 per million output tokens, the token-only estimate is **$0.0055725 for all fifteen requests**. This is an estimate rather than a billing invoice, and excludes any account-specific premiums. Time, accuracy and promotion safety are the material tradeoffs here.

## Installation and operational tradeoffs

The packages were installed into disposable isolated environments, not the application's requirements:

- Visions 0.8.2 used Python 3.14 with pandas 3.0.6 and numpy 2.5.3. Its `CompleteSet` required the image-path and geometry extras (`visions[type-geometry,type-image-path]==0.8.2`) to satisfy imports. This adds dependencies even though these CSVs do not need image or geometry inference.
- DataProfiler 0.14.0 used Python 3.11.15, TensorFlow 2.21.0, pandas 2.3.3 and numpy 2.4.6. The app's Python 3.14 environment could not support that ML dependency combination. The bundled structured model loaded and actually predicted; Keras warned that optimizer variables differed between the stored model and installed optimizer. Inference completed, but the warning and older-runtime requirement need packaging validation before integration. Seventy-three nonempty columns triggered predictions per run; the empty taxi field was retained as unknown.
- Luna used the application's configured OpenAI key and exact `gpt-6-luna` model. No keys are included in results. Its small client RSS does not include remote model memory or guarantee offline operation.

Complete installed package lists are captured alongside results as `visions-environment.txt` and `dataprofiler-environment.txt`.

## Validation and reproducibility

Cleanup note: the rejected library/batching experiment runners are archived in `/private/tmp/analytico-retired-experiments-2026-10-03.tar.gz`; restore them before using their historical reproduction commands below. Raw sample snapshots, per-trial responses and screenshots were removed from the source tree after this experiment. Compact `comparison.json` and dependency freezes remain. The original raw files are saved in `/private/tmp/analytico-benchmark-archive-2026-10-03.tar.gz`; restore those files before rescoring historical results, or regenerate inputs and runs using the commands below. Archive contents are diagnostic evidence, not required application data.

Three focused benchmark tests passed: response index coverage, physical/domain compatibility, and prompt/selection isolation. A separate review replayed all 210 selected-column decisions across the fifteen live records and confirmed that the recorded merge followed its declared gates. This validates the experimental contract, not the truth of model self-confidence. No application source values, production role logic, dependencies or UI were changed by this benchmark.

The scorer records the input snapshot and independent role-manifest SHA-256 hashes; each input dataset also records its complete source-file SHA-256. `comparison.json` includes each trial's per-dataset results, all disagreements, abstention coverage, assigned-role precision and timing summaries. Native labels, distributions, individual proposals and decisions remain in the underlying library/Luna artifacts.

Run from the repository root, with the real fixtures available and each library environment installed from its captured package list:

```sh
backend/venv/bin/python backend/benchmarks/library_inputs.py --output backend/benchmarks/results/library-comparison-2026-10-03/inputs.json
/private/tmp/analytico-visions-venv/bin/python backend/benchmarks/library_visions.py --input backend/benchmarks/results/library-comparison-2026-10-03/inputs.json --repeats 3
/private/tmp/analytico-dataprofiler-venv/bin/python backend/benchmarks/library_dataprofiler.py --input backend/benchmarks/results/library-comparison-2026-10-03/inputs.json --output backend/benchmarks/results/library-comparison-2026-10-03/dataprofiler.json --repeats 3
backend/venv/bin/python backend/benchmarks/semantic_hybrid.py --live --input backend/benchmarks/results/library-comparison-2026-10-03/inputs.json --output backend/benchmarks/results/library-comparison-2026-10-03/luna-hybrid.json --repeats 3
backend/venv/bin/python backend/benchmarks/score_library_comparison.py
backend/venv/bin/python -m unittest discover -s backend/tests -p test_semantic_hybrid_benchmark.py
```

The live command requires the existing configured key; it makes new billable requests. Input preparation can reuse source-hash-matching snapshots, so delete or use a new output path when benchmarking changed ingestion policy. The old baseline remains a recorded measurement. The extra DataProfiler high-water probe is preserved separately; it does not replace any formal repetition.

## Limits and next step

This is a five-dataset, 74-column test of semantic roles, not a universal accuracy claim, a casting benchmark, a weighted-aggregation benchmark, or a chart-answer evaluation. Library adapters and the supported analytical-role mapping are explicit. The library rows cannot be compared with their published entity-classification accuracies. Numeric/storage-type agreement is also not semantic correctness. Luna has richer header/table context than the value-only native labeler, which is a deployment difference rather than an equal-input model contest.

The next justified change is optional background Luna enrichment with conservative merge rules, keeping full-column conversion validation and source preservation local. Preserve established identifier usage and user decisions, separate calendar components from chart grouping choices, and display disputed suggestions through Review data. Evaluate that narrower policy on fresh held-out datasets before allowing automatic overwrites. Adding either library to the main ingestion path is not supported by these results.
