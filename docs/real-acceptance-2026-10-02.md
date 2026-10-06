# Real-data acceptance: first iteration

Executed October 2–3, 2026. Direction: [product goal](product-goal.md). Raw evidence: [benchmark artifacts](../backend/benchmarks/results/real-acceptance-2026-10-02/). This is a smoke evaluation, not a production accuracy or p95 guarantee.

## Decision

Keep GPT-6 Luna, the existing interpretation prompt, and optional background analysis. Do not promote the two stricter prompt experiments or experimental batching. Both stricter prompts reduced useful coverage on real data. The highest-payoff accepted correction is preventing a verified row-position sequence from appearing as an automatically summed business measure, with full-column evidence supplied to Luna.

The next accuracy milestone should separate role, unit, parsing and aggregation decisions, ground them in optional source definitions, and explicitly clarify unsupported business meaning. More prompting alone did not resolve opaque names. A faster classifier should be considered only after this decision contract and independent evaluation improve.

## Real workloads and ground truth

The [versioned manifest](../backend/evals/real_dataset_manifest.json) records documented roles, allowed code-role alternatives, queries and sources. Model requests receive names, at most 12 spaced sample values, peer names and compact statistics; they never receive evaluation labels or source codebooks.

| Dataset | Source | Rows × columns | Characteristics |
|---|---|---:|---|
| Bike sharing | [UCI](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset), complete hour.csv | 17,379 × 17 | Opaque codes, row index, counts, normalized measurements |
| Bank marketing | [UCI](https://archive.ics.uci.edu/dataset/222/bank+marketing), complete bank-full.csv | 45,211 × 17 | Semicolon CSV, count versus category ambiguity, coded calendar fields |
| Online retail | [UCI](https://archive.ics.uci.edu/dataset/352/online+retail), entire workbook exported to CSV | 541,909 × 8 | Negative return quantities, missing customers/descriptions, mixed invoice identifiers |
| NYC green taxi | Existing repository public dataset | 1,068,755 × 20 | Sparse columns, dates, fares, distances and numeric category codes |

No rows were filtered to simplify the fixtures. The retail export changes the container from XLSX to CSV; its complete worksheet remains intact. SHA-256 fingerprints and machine metadata are in the JSON. An independent streaming CSV/Decimal oracle checks complete row counts, missing counts and grouped sums/means, including negatives and null groups. Numeric comparison uses relative tolerance 1e-9 and absolute tolerance 1e-6; the product still uses floating-point arithmetic.

## Measured performance

Arm64 macOS, separate worker processes, two local-ingestion trials per dataset per accepted run. Values are medians. Local readiness is stage plus confirm, with background enrichment scheduling disabled for timing. It excludes browser transfer/render. RSS is process lifetime peak, including the service benchmark and reference checks, not system-wide or concurrent-session memory.

| Dataset | Local ready before → after | Peak RSS before → after | After aggregate API cold / warm | After column AI wall time |
|---|---:|---:|---:|---:|
| Bike | 0.370 → 0.371 s | 185.8 → 185.8 MB | 9.1 / 1.2 ms | 5.80 s |
| Bank | 1.016 → 0.997 s | 227.6 → 227.3 MB | 11.9 / 1.4 ms | 5.19 s |
| Retail | 3.468 → 3.533 s | 289.9 → 294.0 MB | 11.1 / 1.3 ms | 3.85 s |
| Taxi | 13.417 → 14.483 s | 358.9 → 358.7 MB | 15.7 / 1.2 ms | 6.50 s |

AI times are one uncached run per dataset using four parallel column requests and GPT-6 Luna. They exclude business-summary generation. They are background work and must not be added to local readiness as a blocking ingestion step. Aggregate times use the in-process HTTP test client, not browser question-to-render time. Small readiness changes do not establish speedups; taxi readiness remains above the proposed 8-second median target. These changes prioritize correctness, not ingestion speed.

## Accuracy and failed experiments

Direct comparison with the current local, non-Luna classifier on the same recorded columns: bike 15/17, bank 16/17, retail 7/8, taxi 15/20, totaling 53/62 (85.5%). Luna returned 58/62 (93.5%): five additional correct roles overall, including four additional taxi roles. This is a meaningful improvement on these cases, not a reconstruction of the original repository's historical classifier. The baseline below is an earlier Luna run, not the non-AI baseline. Luna also loses two locally correct roles: bank `campaign` becomes incorrect and taxi `extra` has a provider failure. These proposals remain advisory until review; their score is not evidence that the active automatic charts already use them.

| Dataset | Baseline role agreement | Accepted run role agreement |
|---|---:|---:|
| Bike | 14/17 | 16/17 |
| Bank | 15/17 | 16/17 |
| Retail | 8/8 | 8/8 |
| Taxi | 19/20 | 18/20 |
| Total | 56/62 (90.3%) | 58/62 (93.5%) |

The accepted run includes two cautious abstentions (`instant`, an opaque bike index, and taxi `ehail_fee`, an entirely empty column), one provider failure (`extra`), and one confidently incorrect role (`campaign`, a bank contact count called categorical). All four remain in the denominator. Correct role agreement does not prove correct units, aggregations or parsing. Different single live runs confound input changes with model/network variation; the difference is descriptive and is not a demonstrated causal accuracy gain.

Both runs passed the 12 selected question cases: eight chart plans with independently verified values and four clarification cases (prediction, weighted average, derived retail revenue, ambiguous taxi revenue). These few explicit questions cannot establish universal reliability.

The existing 30-case regression baseline returned 25 complete decision matches, 30 role matches, caught all 11 required clarifications, and unnecessarily clarified one other case. The strict candidate dropped complete matches to 23 and abstained heavily on bike and bank; a narrowed candidate still returned only 7/17 bike role matches. Both were rejected. Their `after*` and `candidate-refined*` artifacts are experimental failures, not the promoted production version. The original v3 prompt is unchanged.

## Accepted changes and validation

- Both pandas and DuckDB verify the entire integer column before flagging a dense 0/1-based row-position sequence. Automatic profiles/charts/suggestions exclude it. Values, types and explicit queries remain available; an explicit reviewed metric role restores automatic inclusion. This may suppress a legitimate count-up measure from unsolicited suggestions, so it deliberately does not relabel or delete the column.
- Luna receives existing full-column distinct counts and that sequence flag alongside bounded source samples. Statistics describe the parsed analytics view; samples describe the source values.
- The benchmark exits unsuccessfully if any dataset fails, while retaining sanitized failure evidence. The initial taxi oracle mismatch was null-key representation (`None` versus empty string), corrected in the oracle. It was not an engine arithmetic defect.
- The bike default no longer sums `instant`. Its next automatic selection, mean hour by date, is still weak semantically. This is concrete evidence that further chart-selection/meaning work remains.

Backend validation: 217 tests. Frontend helper checks include completed, uncertain and absent AI decisions. Type checking and lint also run; see the recorded validation result for exit status.

## Codex in-app browser evidence

Only Codex's in-app browser was used. Bank uploaded with column AI off; charting and optional review remained available. Bike uploaded with AI on, rendered its chart and generated a weekday count chart. Full retail uploaded with AI on, rendered 541,909 rows, completed 8/8 column reviews, and opened Review data only on request. Uncertain proposals were visibly disabled rather than crashing React. No console errors were captured for these flows.

The first retail question timed out at the provider and returned HTTP 502. The UI showed the timeout and restored the existing chart. Retry generated “Total Quantity by Country” without another upload. This real failure is retained as a failure, not counted as a first-attempt success. [Browser screenshot](../backend/benchmarks/results/real-acceptance-2026-10-02/retail-browser.png).

Codex's large-file chooser/setFiles action took 503 seconds for the 47.8 MB retail CSV. The server recorded confirmation processing of 3.71 seconds. The chooser measurement includes the automation file bridge, transfer and waits; its delay has not been localized. It is not a valid 503-second parser measurement or a production upload SLA. An ordinary user-upload measurement is still needed before asserting end-to-end large-file readiness.

## Reproduce

From the repository root, download the two CSV archives without additional packages:

```sh
backend/venv/bin/python backend/benchmarks/fetch_real_fixtures.py
```

For retail, run the same fetch script with `--datasets retail` using a Python environment with openpyxl. The workbook is exported without filtering. The existing taxi CSV path is specified in the manifest. Once fixtures exist:

```sh
backend/venv/bin/python backend/benchmarks/real_acceptance.py --output /private/tmp/acceptance-local.json
backend/venv/bin/python backend/benchmarks/real_acceptance.py --live --output /private/tmp/acceptance-live.json
```

`--live` makes paid requests with the configured OpenAI key and asserts GPT-6 Luna. It is deliberately opt-in. Without `--live`, no interpretation/query calls are made. Browser reproduction: open localhost:3000 in Codex; upload bank with the switch off; upload bike/retail with it on; wait for the chart independently of AI; open Review data; ask the manifest queries; inspect console errors. No Chrome/Brave or external Playwright runner is required.

## Remaining acceptance work

This iteration is complete, but the release milestone is not. Still required: repeated held-out domains and question paraphrases; explicit missing-key browser coverage; reload/interruption tests during active uploads; wide real data; field-level interpretation scoring; ordinary large-file transfer/render measurements; at least 100 latency observations before p95 claims; and correcting semantically weak automatic charts. Existing parsing/provider regression tests are useful but do not substitute for those browser and live-model cases. Follow [OpenAI's evaluation guidance](https://developers.openai.com/api/docs/guides/evaluation-best-practices) by expanding independent cases and retaining errors rather than tuning to a single dataset.
