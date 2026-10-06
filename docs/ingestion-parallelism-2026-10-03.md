# Ingestion parallelism experiments — 2026-10-03

## Recommendation

Four DuckDB threads are the simplest promising change on this eight-logical-CPU machine: about 17% faster taxi preparation and 16% faster retail preparation, while preserving the checks. They increase peak ingestion RSS. The four-thread default was subsequently promoted with CPU capping and an explicit local override; see the promotion section below. Eight threads add little further speed. Concurrent validation batches help some files but add scheduling/connection complexity. Process workers are not a consistent win and consume much more memory.

Proof-based pruning also preserved results: it first counts non-missing and plain-number values across every row. Only when all non-missing values satisfy the existing plain-number grammar does it substitute logically implied format counts. Integer range, leading-zero, precision and date checks still run. This is an experiment, not a sample-based shortcut or production change.

## Local performance

Three fresh-process runs per file/configuration; all eight original full public files from the frozen acceptance manifest. Modes force the disk engine to compare the same implementation. Small files currently use pandas in production, so their forced-disk improvements should not be advertised as current-app speedups. Randomized ordering for the five parallelism modes; the pruning experiment ran afterward as a separate three-repeat sweep. Filesystem caches were warm.

| Dataset | Two threads | Four threads | Eight threads | Concurrent thread batches | Process batches | Proven plain pruning |
|---|---:|---:|---:|---:|---:|---:|
| bike | 0.382s | 0.380s | 0.380s | 0.274s | 0.352s | 0.224s |
| bank | 0.996s | 0.986s | 0.989s | 0.684s | 0.632s | 0.846s |
| retail | 3.355s | 2.817s | 2.768s | 2.944s | 3.634s | 3.241s |
| taxi | 13.033s | 10.838s | 10.667s | 12.372s | 13.305s | 11.801s |
| wine | 0.128s | 0.127s | 0.129s | 0.110s | 0.226s | 0.083s |
| diamonds | 0.704s | 0.701s | 0.701s | 0.527s | 0.508s | 0.501s |
| penguins | 0.068s | 0.066s | 0.068s | 0.064s | 0.196s | 0.061s |
| tips | 0.068s | 0.069s | 0.069s | 0.067s | 0.195s | 0.064s |

| Taxi configuration | Change from two threads | Peak ingestion memory |
|---|---:|---:|
| threads2 | +0.0% elapsed time | 285 MiB |
| threads4 | -16.8% elapsed time | 345 MiB |
| threads8 | -18.2% elapsed time | 418 MiB |
| batches_threads2 | -5.1% elapsed time | 303 MiB |
| batches_processes2 | +2.1% elapsed time | 418 MiB |
| proven_plain_pruning | -9.4% elapsed time | 279 MiB |

Memory uses process high-water RSS for single-process modes and sampled aggregate process-tree RSS for multiprocessing. The latter is an observed lower bound, especially for tiny files. RSS includes runtime/library overhead and is not the DuckDB 64MB execution budget. The multiprocessing candidate gives each of two readers 64MB; it is not equal in total resource budget to the thread candidate. Detailed phase medians, ranges and both RSS measurements are in the compact summary.

Concurrent validation reuses the exact original statistics queries in batches of four columns. Thread readers use separate DuckDB cursors. Process readers require checkpointing and closing the parent connection, two read-only worker processes, and reopening the parent afterward; all that overhead is included. This tests process-based column validation, not every possible Python multiprocessing design.

## AI overlap

The current arm sends the existing full-schema request after local preparation. The experimental arm sends the same structured-output contract with original headers, at most twelve values from the first 100 source rows, and explicitly provisional/sample-only metadata. It starts before local ingestion. Its response is then passed through the production compatibility validator using the completed full profile; no roles apply before local validation. This is experimental ingestion behavior, not a new UI gate. Both arms use exact GPT-6 Luna, with independent business-summary enrichment disabled.

Two trials per file/arm, 32 completed live schema calls. Current final role coverage: **189/196 correct (96.4%), seven unknown, zero wrong applied roles**. Early-sample coverage: **181/196 correct (92.3%), fifteen unknown, zero wrong applied roles**. These are codebook scores in this test set, not calibrated probabilities or universal guarantees. Labels were not separately scored.

| Dataset | Current wait after local | Early wait after local | Current AI roles ready from start | Early AI roles ready from start |
|---|---:|---:|---:|---:|
| bike | 5.414s | 4.650s | 5.734s | 4.978s |
| bank | 4.902s | 4.480s | 5.810s | 5.359s |
| retail | 2.646s | 0.467s | 6.055s | 3.801s |
| taxi | 5.805s | 0.054s | 19.936s | 13.269s |
| wine | 3.667s | 4.002s | 3.756s | 4.096s |
| diamonds | 3.364s | 3.133s | 3.946s | 3.711s |
| penguins | 2.798s | 2.790s | 2.842s | 2.839s |
| tips | 2.708s | 2.519s | 2.777s | 2.576s |

The first chart already appears after local preparation in both arms. Earlier AI improves role readiness, not the local upload-to-first-chart time. Retail and taxi retained full-profile coverage in this small sample; early requests missed additional interpretations in bank, bike, tips and wine. Do not promote this sample-only approach across the board based on two trials. A large-file-only experiment or selective full-profile follow-up may be worth testing, but adds behavior and potentially another request. Network variation and two repeats make timing exploratory; the first few AI cases overlapped the short regression suite.

## Correctness and reproduction

- All **144** local imports matched baseline metadata and two full-table checksums (including source row ordinals); all retained source hashes and row counts matched the frozen manifest. Checksums establish equality to the existing baseline, not an independent proof that every baseline interpretation is right.
- **35** relevant regression tests passed under each of six configurations (**210 passes**), covering native/fallback loading, malformed inputs, exact integers, mixed currencies, nulls and overall aggregation. The unchanged production suite passed **242 tests**.
- Full raw records stay in `/private/tmp/analytico-parallelism-2026-10-03`, `/private/tmp/analytico-parallelism-pruning-2026-10-03` and `/private/tmp/analytico-ai-overlap-2026-10-03.json`. Only a compact summary is retained in the repo.

Run `backend/venv/bin/python backend/benchmarks/ingestion_parallelism.py --repeats 3` for all local experiments. The first five configurations were run separately from pruning in this recorded comparison. Run `backend/venv/bin/python backend/benchmarks/ai_ingestion_overlap.py --live --repeats 2` for paid AI overlap checks. Public fixtures must already exist at the paths recorded by `backend/evals/fresh_accuracy_gold_2026-10-03.json`. The benchmark itself changed no production settings; the subsequent promotion is documented below.

[Compact results](../backend/benchmarks/results/parallelism-2026-10-03/summary.json)

## Promotion after the benchmark

The accepted production default is now up to four DuckDB threads per dataset, capped to `os.cpu_count()`. When CPU capacity is unavailable it falls back to two. `ANALYTICO_DUCKDB_THREADS` accepts 1–8 and is also capped to available CPUs; invalid values fail as configuration errors before allocating dataset files. The DuckDB memory budget remains 64MB. Existing validation and full-profile Luna analysis are retained. The table above records the historical experimental comparison against the two-thread baseline.

Promotion validation: 245 production backend tests passed. Twelve deterministic overall-aggregation checks passed against independent complete-source Decimal calculations on tips, diamonds and the million-row taxi file using the new default. Detailed verification is `/private/tmp/four-thread-real-verification.log`.
