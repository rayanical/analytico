# Fresh ingestion and query accuracy benchmark

Three fresh repetitions compare local deterministic inference with production automatic Luna schema enrichment. Eight real datasets and 20 frozen questions cover explicit grouped charts and clarifications. The gold was frozen before any live calls; held-out CSV bytes are pinned in the checked-in manifest.

- Datasets: 8; ingestion runs: 48/48.
- Queries: 40/40; correct: 38 (0.950). Failures: `{"wrong_answer_kind": 2}`.
- Role results: local 243/294 correct, 51 unknown, 0 wrong; auto Luna 284/294 correct, 10 unknown, 0 wrong.
- All-column schema responses completed in 24/24 automatic runs. Luna resolved 41 baseline unknown roles; no applied role disagreed with the accepted codebook rubric.
- 85 readable AI display labels were applied across the first automatic repeat, retained in the compact label-map output for review.
- Retained source SHA, row count, and full missing counts matched on 48/48 ingestion runs.
- All 8/8 full-source hashes also match the frozen gold manifest.
- Live planner request latency includes 44 calls: 40 scored cases plus four taxi rescoring calls. The single ambiguous ‘best tips’ prompt was asked in both arms and answered with a chart both times, accounting for the two wrong-answer-kind cases.

## Results by arm

| Arm | Applied roles correct / evaluated | Unknown | Wrong | Role accuracy | Query score | Median planner (s) | Median aggregate (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| local only | 243/294 | 51 | 0 | 0.827 | 19/20 (0.950) | 1.607 | 0.007 |
| auto Luna schema | 284/294 | 10 | 0 | 0.966 | 19/20 (0.950) | 1.685 | 0.007 |

## Dataset-level results

| Dataset | Arm | Ready median (s) | Schema wait median (s) | Luna call median (s) | Correct / unknown / wrong roles | Correct queries |
|---|---|---:|---:|---:|---:|---:|
| bank | auto_luna_schema | 0.969 | 4.635 | 4.546 | 47 / 4 / 0 | 3/3 |
| bank | local_only | 0.984 | — | — | 42 / 9 / 0 | 3/3 |
| bike | auto_luna_schema | 0.365 | 5.114 | 5.026 | 48 / 3 / 0 | 3/3 |
| bike | local_only | 0.372 | — | — | 21 / 30 / 0 | 3/3 |
| diamonds | auto_luna_schema | 0.608 | 3.406 | 3.386 | 30 / 0 / 0 | 2/2 |
| diamonds | local_only | 0.624 | — | — | 30 / 0 / 0 | 2/2 |
| penguins | auto_luna_schema | 0.065 | 2.997 | 2.942 | 21 / 0 / 0 | 2/2 |
| penguins | local_only | 0.074 | — | — | 18 / 3 / 0 | 2/2 |
| retail | auto_luna_schema | 3.365 | 2.775 | 2.709 | 24 / 0 / 0 | 3/3 |
| retail | local_only | 3.414 | — | — | 24 / 0 / 0 | 3/3 |
| taxi | auto_luna_schema | 13.462 | 6.116 | 6.004 | 57 / 3 / 0 | 3/3 |
| taxi | local_only | 13.652 | — | — | 57 / 3 / 0 | 3/3 |
| tips | auto_luna_schema | 0.070 | 2.567 | 2.548 | 21 / 0 / 0 | 2/3 |
| tips | local_only | 0.091 | — | — | 18 / 3 / 0 | 2/3 |
| wine | auto_luna_schema | 0.113 | 4.186 | 4.086 | 36 / 0 / 0 | 1/1 |
| wine | local_only | 0.119 | — | — | 33 / 3 / 0 | 1/1 |

## Ingestion stage medians

Seconds across the three repetitions. For disk rows, `disk_ingestion` is the enclosing total; the component columns below do not include or add that wrapper.

| Dataset | Arm | Stage + preview | Confirm total | Source copy | Structure validation | Engine setup | CSV parse/load | Type/profile plan | Typed materialize | Column profile | Chart metadata | Storage |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bank | auto_luna_schema | 0.093 | 0.874 | 0.002 | — | — | 0.115 | 0.738 | — | 0.001 | 0.000 | 0.014 |
| bank | local_only | 0.108 | 0.879 | 0.002 | — | — | 0.116 | 0.745 | — | 0.001 | 0.000 | 0.014 |
| bike | auto_luna_schema | 0.058 | 0.305 | 0.001 | — | — | 0.040 | 0.253 | — | 0.001 | 0.000 | 0.006 |
| bike | local_only | 0.067 | 0.304 | 0.001 | — | — | 0.040 | 0.255 | — | 0.001 | 0.000 | 0.006 |
| diamonds | auto_luna_schema | 0.046 | 0.559 | 0.002 | — | — | 0.082 | 0.463 | — | 0.001 | 0.000 | 0.010 |
| diamonds | local_only | 0.067 | 0.558 | 0.002 | — | — | 0.082 | 0.463 | — | 0.001 | 0.000 | 0.010 |
| penguins | auto_luna_schema | 0.026 | 0.038 | 0.000 | — | — | 0.001 | 0.034 | — | 0.000 | 0.000 | 0.000 |
| penguins | local_only | 0.036 | 0.037 | 0.000 | — | — | 0.001 | 0.034 | — | 0.000 | 0.000 | 0.000 |
| retail | auto_luna_schema | 0.041 | 3.322 | 0.011 | 0.675 | 0.005 | 0.681 | 1.510 | 0.368 | 0.059 | 0.005 | 0.000 |
| retail | local_only | 0.054 | 3.359 | 0.011 | 0.687 | 0.005 | 0.688 | 1.531 | 0.368 | 0.060 | 0.005 | 0.000 |
| taxi | auto_luna_schema | 0.101 | 13.359 | 0.032 | 2.173 | 0.005 | 3.323 | 6.150 | 1.306 | 0.336 | 0.014 | 0.000 |
| taxi | local_only | 0.103 | 13.549 | 0.033 | 2.348 | 0.005 | 3.385 | 6.058 | 1.326 | 0.334 | 0.014 | 0.000 |
| tips | auto_luna_schema | 0.026 | 0.042 | 0.000 | — | — | 0.001 | 0.040 | — | 0.000 | 0.000 | 0.000 |
| tips | local_only | 0.045 | 0.046 | 0.000 | — | — | 0.001 | 0.043 | — | 0.000 | 0.000 | 0.000 |
| wine | auto_luna_schema | 0.036 | 0.074 | 0.000 | — | — | 0.010 | 0.061 | — | 0.000 | 0.000 | 0.001 |
| wine | local_only | 0.046 | 0.072 | 0.000 | — | — | 0.010 | 0.060 | — | 0.001 | 0.000 | 0.001 |

## Accuracy and measurement scope

Each question’s plan must return the expected kind, exact axes, and aggregation. Chart results are compared over the full source to an independent Python `csv.DictReader`/`Decimal` oracle, including null X-axis groups. Clarification cases score only answer kind. Two “best tips” prompts remained ambiguous under the frozen policy but produced charts; those are answer-kind errors, separate from numeric mismatches. Four taxi chart requests were repeated once per arm after the oracle was corrected; those requests are recorded separately and do not increase the 40 scored question count.

Source integrity checks cover source SHA256 after retention, complete row count, and full missing counts. The chart oracle validates selected X/Y columns only and does not establish cell-by-cell correctness for every typed value in unrelated columns. No blanket ingestion accuracy is inferred from these checks.

The local-ready timer spans `stage_import` and `confirm_import`; the ingestion telemetry separately reports parser/source-copy, validation, type planning, materialization, profile, chart metadata, and storage phases. Automatic enrichment is asynchronous, so observed queue wait and the timed `analyze_schema` call are reported separately. Query total, planner call, and `run_aggregate` timings are separate. All timing summaries use medians at n=3; no p95 is reported.

Current RSS is a point-in-time sample when available; high-water RSS is lifetime-wide across the benchmark process and repetitions, not per-import. Later repetitions run with a warmed OS file cache. These are repeated local measurements on one host, not cold-storage timings.

The compact JSON includes the first automatic repeat’s applied label maps. Full per-trial roles, all label maps and label-evidence proposals are in `/private/tmp/analytico-fresh-accuracy-2026-10-03-raw.json` for independent review.

Frozen-gold SHA256: `7d52429b8b5c3e1d9517ed03ce292ec34600dd1ca26a8be659064d51420f2b60`. Schema prompt version: `schema-role-label-v2`. Schema prompt, JSON schema, query prompt, and production code fingerprints are recorded in `summary.json` from before the root prompt update.

Reproduce the baseline: `backend/venv/bin/python backend/benchmarks/fresh_real_accuracy.py --live`. Rebuild only summary/report from saved results: `backend/venv/bin/python backend/benchmarks/fresh_real_accuracy.py --finalize-existing`.

## Data sources and codebooks

- **bike** — 17,379 rows, 17 columns. [Source and codebook](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset); SHA256 `e03de4ee4ef4dc376ac6e04bf829673c6269e8eba5c60fa121640fa2f829504f`.
- **bank** — 45,211 rows, 17 columns. [Source and codebook](https://archive.ics.uci.edu/dataset/222/bank+marketing); SHA256 `d1513ec63b385506f7cfce9f2c5caa9fe99e7ba4e8c3fa264b3aaf0f849ed32d`.
- **retail** — 541,909 rows, 8 columns. [Source and codebook](https://archive.ics.uci.edu/dataset/352/online+retail); SHA256 `50e9f56b0ee4bd0662a397c68295f48945972135f57322cd5208f4f809eb67ac`.
- **taxi** — 1,068,755 rows, 20 columns. [Source and codebook](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page); SHA256 `b217661724e4d6270063d1eb460dca49aef1187383d6ce30025405d7bb4945c1`.
- **wine** — 4,898 rows, 12 columns. [Source and codebook](https://archive.ics.uci.edu/dataset/186/wine+quality); SHA256 `76c3f809815c17c07212622f776311faeb31e87610d52c26d87d6e361b169836`.
- **diamonds** — 53,940 rows, 10 columns. [Source and codebook](https://ggplot2.tidyverse.org/reference/diamonds.html); SHA256 `9574730b03aba241d899c4a97511c5061b19358fab89510774fb6c24168345c4`.
- **penguins** — 344 rows, 7 columns. [Source and codebook](https://allisonhorst.github.io/palmerpenguins/reference/penguins.html); SHA256 `e07636bd8af74260099ea2f8678e2eabbf35def579940cc76f67061ee16c06c1`.
- **tips** — 244 rows, 7 columns. [Source and codebook](https://search.r-project.org/CRAN/refmans/reshape2/html/tips.html); SHA256 `e54cc4d2ce1bff65d32ca60b3e4b802e06bde1d7e7caf6f796f6bf7370e863b0`.

Held-out `diamonds`, `penguins`, and `tips` CSVs are from seaborn-data commit `71e2436a092d714350de0fc409ca8a8714e7e78f`. Its [README](https://github.com/mwaskom/seaborn-data#seaborn-data) describes the files as convenience samples and cautions that some differ from canonical sources; the linked codebooks define role labels. The seaborn penguins CSV omits the documented `year` variable, so this seven-column snapshot does not score it.

## Shipped labels and answer follow-up

Readable display names now share the GPT-6 Luna role-analysis request. Stable column keys and original headers are retained. Labels can apply independently of role certainty; empty, oversized, control-character and colliding automatic labels are rejected. Labels are cosmetic aliases, not validated business definitions. Source units are not inferred. The initial run produced 85 non-null applied labels across 98 columns in its first repetitions; this is coverage, not an independently calibrated semantic-accuracy score. Review data exposes editable names and source headers. Name-only edits update metadata without re-ingestion, retain active AI roles, and use a new version to reject stale user edits. Background updates preserve the displayed chart.

The initial frozen query baseline remains **19/20 per arm (38/40 total)**. Both misses silently interpreted “best tips” as average tips. A generic planner instruction now asks for an objective when totals, averages, frequencies or rates would answer materially different questions; it contains no dataset or column-name exceptions.

After that fix, a separate live pass of the same frozen questions scored **20/20 per arm (40/40)**. A follow-up set scored **9/9 per arm (18/18)**: the observed ambiguous prompt repeated three times, three new ambiguous comparisons in other datasets, and three new explicit sums/averages with independent full-source value checks. This is a regression improvement, not an independent new 100% baseline. The new-question subset is small, and repeated prompts are not independent examples. These checks do not establish universal correctness, filter coverage, arbitrary business definitions, conversational follow-up support or exact accounting arithmetic.

The compact [answer follow-up summary](../backend/benchmarks/results/fresh-accuracy-2026-10-03/answer-followup-summary.json) records frozen cases, prompt hashes, counts and timings separately from the initial ingestion baseline. Current-source reproduction of the main runner uses the updated query prompt and should be recorded as a new run rather than overwriting the interpretation of the historical 95% baseline. Detailed follow-up records remain in `/private/tmp`.

All 236 backend tests, frontend helper tests, TypeScript checks and diff whitespace validation passed. The Codex in-app browser verified an immediate chart while AI was pending, readable background labels on the retained chart, optional editable Review data, source-name tooltips, and no captured browser console errors.

The largest measured remaining ingestion costs are taxi type/validation planning (**6.15 s**), CSV loading (**3.32 s**), source structure validation (**2.17 s**) and typed materialization (**1.31 s**). Optimize redundant scans while retaining full-column safety checks before adding another AI ingestion dependency.
