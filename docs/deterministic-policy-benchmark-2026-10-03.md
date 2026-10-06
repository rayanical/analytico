# Deterministic column policy: results and rule audit

Measured 2026-10-03. Implemented the separation recommended by [the open-source research](open-source-inference-research-2026-10-03.md): physical type, proposed role, evidence, numeric/grouping capabilities and automatic aggregation are distinct. One policy serves pandas and DuckDB. Review remains optional. GPT-6 Luna remains unchanged and its suggestions remain advisory.

## Accuracy: fewer incorrect assignments, lower coverage

These are complete public datasets with source-codebook labels, not fabricated performance fixtures. Compare the immediate pre-change deterministic baseline with the final policy on the same four datasets:

| Dataset | Columns | Correct before | Correct after | Wrong assigned before → after | Unknown after |
|---|---:|---:|---:|---:|---:|
| UCI Bike Sharing, hourly | 17 | 15 | 7 | 2 → 0 | 10 |
| UCI Bank Marketing, bank-full | 17 | 16 | 14 | 1 → 0 | 3 |
| UCI Online Retail | 8 | 7 | 8 | 1 → 0 | 0 |
| NYC Green Taxi, complete repository CSV | 20 | 15 | 19 | 5 → 0 | 1 |
| **Paired total** | **62** | **53 (85.5%)** | **48 (77.4%)** | **9 → 0** | **14** |

Assigned-role precision rose from 53/62 to 48/48 on this evaluation. Assignment coverage fell from 100% to 77.4%. This is a safer abstention policy, **not an increase in total exact role accuracy**, and zero observed wrong assignments is not a universal guarantee. Several opaque bike names were previously labeled correctly by coincidence and are now unknown. The remaining unknowns include `casual`, `registered`, bank contact counts and the entirely empty taxi `ehail_fee`.

The held-out [UCI white wine quality dataset](https://archive.ics.uci.edu/dataset/186/wine+quality), 4,898 rows × 12 columns, scored 11 correct, one unknown, zero wrong assigned roles. Its ordered quality score remains unknown; there is no ordinal role in the current contract. It was not added to the paired denominator. Source distributions, negative retail returns and missing values were retained. Gold labels and codebooks were not supplied to model requests.

All four automatic chart measures now match a measure role in the source labels, versus three before. Defaults use an explicitly described arithmetic mean unless an accepted definition supplies another aggregation. The two previous automatic SUM defaults became means; this avoids assuming additivity but does not establish business meaning or a weighted average. Explicit user SUM requests still work. If no automatic measure is supported, a frequency chart counts non-missing values of its named field.

The previous Luna result was 58/62 (93.5%) on these four datasets. That is historical context, not a rerun or a measured combined-system score. Unknown numeric columns remain queryable, so abstention does not require a review dialog before charting. Source metadata or cautiously validated Luna suggestions remain useful for opaque codes and business definitions.

## Performance

Three isolated trials per dataset on this arm64 macOS machine; median local staging + confirmation, with background AI disabled. Includes complete-column validation and storage. Excludes file transfer, browser rendering and network AI. Peak RSS is the entire worker process high-water mark, including independent reference checks; it is not retained dataset memory or concurrent-user capacity.

| Dataset / complete rows | Local ready before → after | Peak RSS before → after |
|---|---:|---:|
| Bike / 17,379 | 0.376s → 0.384s | 170.4 → 170.6 MB |
| Bank / 45,211 | 0.966s → 1.002s | 220.9 → 221.5 MB |
| Retail / 541,909 | 3.396s → 3.353s | 270.3 → 270.8 MB |
| Taxi / 1,068,755 | 13.378s → 13.155s | 325.6 → 331.7 MB |

Wine's final local readiness was 0.115s with 138.3 MB peak RSS; no paired before measurement. The policy change has broadly similar ingestion cost with a small memory increase. Three trials do not support a significant speedup claim or P95 estimate. Taxi still needs performance work to reach the product goal.

## Other hardcoded rules audited

| Rule | Action and reason |
|---|---|
| Numeric distinct count below 20 / uniqueness ratio above 0.5 determines business role | Removed from both engines. Repetition does not prove category or identity. |
| Text distinct count of 50 determines identifier | Removed. Long descriptions and names can repeat or be unique. |
| Prefer grouping cardinality closest to 10 | Removed. Arbitrary diversity does not justify analytical relevance. |
| Separate header aliases for non-additive metrics and automatic sums | Removed in favor of one policy; no dataset-specific `life_exp` exception. |
| Names such as `total`, `value`, `payment` imply currency; `rate`/`ratio` imply percentage | Removed. Units now follow validated source syntax or explicit review, rather than names alone. |
| Unused coded-number detector using max(20, 5% of rows) and 95% integer values | Removed. It was dead code and another incompatible definition of category. |
| At least 50% numeric-looking values required to record a failed numeric parse | Removed. A conversion issue can matter even with one affected value; date-named fields keep date parsing eligibility. |
| Calendar years after 2200 rejected | Replaced with the four-digit calendar domain, 1000–9999. |

Retained deliberately: leading-zero identifier protection, complete-column conversion checks, mixed-currency/ambiguous-date/precision guards, row-position detection and resource limits. Parser regular expressions describe supported syntax; removing them would remove safety validation. Limits on samples, cache sizes, input width and chart output are resource budgets, not semantic classifiers.

Remaining heuristics are explicit in `column_usages.evidence`: key/dimension/quantity name hints, fractional-number candidates and text-grouping defaults. They can still be wrong on unseen data. They do not prove units or additive business metrics and can be overridden. Names such as `account` can still produce false identifier hints. Calendar names plus validated domains are syntax-informed hints, not proof of meaning. Arithmetic means cannot replace source definitions of weighted rates, balances or grain. Further improvements should target these cases with metadata and evals, rather than add another dataset alias.

## Validation and reproducibility

- **223 backend tests passed**, including engine parity, role/aggregation overrides, large calendar years, cardinality invariance, identifiers, frequency fallback and explicit sums of unknown numeric fields. Small artificial edge cases are unit tests, not performance fixtures.
- The real bank `campaign` field now stays unknown on the first 500 rows (five distinct values), the complete file (48 distinct values), and reordered complete rows. Previously the prefix/full roles differed. This checks stability, not improved semantic recall.
- All five complete datasets matched independently streamed row counts, missing counts and Decimal-based grouped aggregate oracles. The final run also proved the retained original CSV bytes match the source hash. The historical baseline did not record byte preservation; it did check rows, null counts and aggregates.
- **13/13 live GPT-6 Luna question cases passed** across the five datasets, plus **2/2 additional unknown-measure questions** on bike and bank. These are selected acceptance cases, not comprehensive semantic coverage. Chart values were checked against the full-source oracle; ambiguity/unsupported-operation cases required clarification.
- Frontend helper checks passed. The Codex in-app browser uploaded the full bike CSV with AI semantic analysis off, rendered a chart without a mandatory review, verified the optional review controls, and successfully charted `registered` despite its unknown role. No captured browser console errors. [Browser proof](../backend/benchmarks/results/deterministic-policy-2026-10-03/bike-browser.png).
- Initial live attempts inside the restricted network environment returned connection failures. Those records (`after.json`, `unknown-measures.json`) are preserved as failed environmental attempts; `after-live-verified.json` and `unknown-measures-network-verified.json` record successful network-enabled reruns. They are not hidden or counted as successful tests.

Raw reports, hashes, code fingerprints and trials: [results directory](../backend/benchmarks/results/deterministic-policy-2026-10-03/). `before.json` is the captured pre-edit baseline, `after-final.json` is the final offline run; intermediate candidate files remain experimental evidence. Live query validation preceded the final small integer-statistic/frequency-axis safeguards, which are covered by the final backend suite and final offline/browser runs.

From `backend/`, with fixtures downloaded using `benchmarks/fetch_real_fixtures.py`:

```sh
venv/bin/python benchmarks/real_acceptance.py --help
venv/bin/python benchmarks/score_column_policy.py --before benchmarks/results/deterministic-policy-2026-10-03/before.json --after benchmarks/results/deterministic-policy-2026-10-03/after-final.json --output benchmarks/results/deterministic-policy-2026-10-03/comparison.json
```

Live checks require an existing configured key and explicit `--live`; fixture manifests document official sources and filenames. No credentials are included in reports. This milestone does not establish production readiness, near-perfect semantic accuracy or a universal upload-to-chart latency guarantee.
