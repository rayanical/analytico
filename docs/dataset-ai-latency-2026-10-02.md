# Dataset ingestion and Luna latency

The strongest immediate improvement is bounded parallel AI requests while keeping ingestion usable before enrichment finishes. A selective hybrid can reduce requests further, but the experimental routing rule loses semantic accuracy on numeric codes. Neither strategy has been enabled in production by this benchmark.

## Measured results

Two isolated-process trials per dataset and strategy on this local ARM64 Mac, Python 3.14. All live requests used GPT-6 Luna, reasoning none, the existing structured-output prompt, and bounded production-style source samples. Application interpretation caching was bypassed. Provider caching, filesystem caching and network conditions were not controlled. These are observed ranges and medians, not latency guarantees.

| Dataset | Size / shape | Local data ready, no AI | Current background AI | Parallel all columns | Selective hybrid |
| --- | --- | ---: | ---: | ---: | ---: |
| Green taxi | 133 MiB; 1,068,755 rows × 20 columns | 21.83 s | +14.90 s; 6–9 columns | +7.86 s; all 20 | +5.18 s; 14 columns |
| Gapminder | 80 KiB; 1,704 rows × 6 columns | 0.054 s | +8.43 s; all 6 | +2.69 s; all 6 | +1.23 s; 2 columns |
| Synthetic wide CSV | 1.94 MiB; 10,000 rows × 50 columns | 0.625 s | +13.94 s; 9–11 columns | +16.20 s; all 50 | +1.46 s; 2 columns |

The AI columns show **additional time after the dataset becomes usable**, including the business summary. Total median time until enrichment completes was 36.36 / 29.57 / 26.61 seconds for taxi (current / parallel / hybrid), 8.49 / 2.74 / 1.29 seconds for Gapminder, and 14.59 / 16.90 / 2.17 seconds for the wide CSV. Each strategy has its own measured ingestion time; the no-AI baseline is not substituted into these totals.

Parallel-all background ranges were 7.52–8.21 seconds for taxi, 2.60–2.78 seconds for Gapminder, and 15.67–16.72 seconds for the wide file. Hybrid ranges were 4.77–5.58, 1.21–1.25 and 1.41–1.52 seconds respectively.

The current implementation runs the summary before sequential column requests, considers at most twelve columns, and checks a twelve-second soft budget between column calls. It can finish with incomplete coverage and can exceed twelve seconds. The experimental parallel worker overlaps summary and column interpretation with at most four requests in flight, considers every column, and waits for its selected requests without that soft budget. It retains the existing per-call timeout. Thus the wide-file current result is **less work**, not evidence that sequential requests outperform parallel requests.

AI-enabled data-ready times remained approximately 21.3–21.9 seconds for taxi, 0.052–0.054 seconds for Gapminder, and 0.645–0.753 seconds for the wide file. AI processing already runs in the background; a blocking AI preview would unnecessarily add the enrichment times above to the upload flow. Bounded sample preparation still happens before the response.

Local staging plus the bounded parsing preview was separately measured twice: taxi 2.45 and 0.33 seconds; Gapminder 0.037 and 0.036 seconds. The taxi first result includes initialization/cache effects; two trials cannot establish a reliable cold-start percentile. Preview timing excludes browser transfer and rendering. Full ingestion timing also excludes browser transfer, multipart parsing and response serialization, and does not include the preceding staging step. The preview is available before the much slower full-file validation and ingestion.

## Why dataset size is not the only factor

Each column request contains at most twelve bounded source values rather than the entire file. More rows primarily increase local validation, conversion and profiling work; more selected columns primarily increase AI requests. Evenly spaced disk samples use the same original lexemes and context as production, extended beyond its twelve-column ceiling only inside the harness. A test verifies the first twelve payloads match production exactly.

Taxi's approximately 22-second data-ready delay is almost entirely in the locally measured disk-ingestion phase. Parallelizing Luna does not remove that delay. A separate ingestion profile is needed before identifying which internal scan or conversion deserves optimization; this measurement groups those operations together. Median isolated-process peak RSS was about 297 MiB with AI off, 326 MiB current, 329 MiB parallel-all, and 326 MiB hybrid. These are lifetime process high-water marks, not per-request allocations or a memory guarantee.

Initial column roles matched across strategies; complete initial profiles matched within 1e-9 floating-point tolerance. Small differences in taxi numeric aggregates came from floating-point reduction order. Every worker also checked that background enrichment left the schema unchanged. No AI decisions were automatically applied.

## Hybrid accuracy tradeoff

The experimental gate sends nonempty categorical, identifier and unknown columns to Luna and skips locally recognized metrics and dates. It contains no taxi-specific names, dataset-specific regex or expected labels. Its inputs depend on the existing local inference, so a mistaken local role can still hide a column from AI. The wide file deliberately contains 48 plain numeric measures and two textual dimensions: its very large request reduction does not represent every wide dataset.

An **offline replay** of the previous thirty saved upload-input predictions tested this gate against the existing semantic-role regression labels:

| Strategy | Model requests | Correct role labels |
| --- | ---: | ---: |
| Local automatic interpretation | 0 | 18/30 (60%) |
| Luna for every column | 30 | 29/30 (96.7%) |
| Experimental selective hybrid | 15 | 27/30 (90%) |

The gate misses a numeric categorical code by accepting its local metric role; it also skips an all-null field whose correct role is unknown. One selected ordinal risk band still receives the wrong AI role. These are small, previously used regression fixtures, not an unseen benchmark or a guarantee for other files. This comparison concerns roles only: local parsing does not supply a complete model decision about units, aggregation and clarification, so hybrid full-decision scores have partial coverage.

The full dataset timing runs have no gold semantic labels. Of 235 live column calls, 69 were actionable `ok` and 166 were `uncertain`; there were no provider failures. All eighteen summaries returned text. An uncertain result is an abstention, not a transport failure or proof of a correct interpretation. The existing prompt often requires unknown aggregation context, which can prevent otherwise useful parsing proposals. Faster request completion does not resolve that issue.

## Recommended implementation order

1. Keep deterministic full-file validation and source preservation before accepting a conversion. Keep the dataset usable while AI enrichment runs, and keep Review columns optional.
2. Replace sequential enrichment with a bounded request pool, overlapping the summary and independent interpretations. Retain dataset-version checks, timeouts and partial progress, and add an explicit coverage indicator for any request or time cap. Parallel requests reduce latency when tasks are independent, as described in [OpenAI's latency guidance](https://developers.openai.com/api/docs/guides/latency-optimization).
3. Separate parsing certainty from semantic role, units and aggregation. Validate any proposed conversion against the complete column, and leave independently unknown metadata unresolved. The current conservative contract blocks too many otherwise useful proposals.
4. Develop and evaluate a broader routing gate before defaulting to selective AI. Numeric identifiers/categories, low-cardinality codes, mixed formats and missing-value ambiguity need coverage. Avoid treating a local `metric` role as proof of semantic correctness. Use fresh labeled datasets and repeated decisions to measure missed columns.

Luna should determine ambiguous meaning or propose a parsing policy from bounded evidence; deterministic code should execute and validate that policy. It cannot eliminate the need to scan the complete column safely. A multi-column structured request may reduce call overhead further, but was not measured here and needs its own accuracy and failure-isolation benchmark.

## Artifacts, cost and reproduction

Results are in `backend/benchmarks/results/dataset-ai-2026-10-02/`: `results.json` contains all 24 trials and safe per-call metadata; `summary.json` contains medians/ranges and consistency checks; `preview.json` contains four local preview measurements; `hybrid-quality.json` contains the offline role regression. Phase timing was added during the run, so the first four taxi trials do not include that field.

Column calls reported 250,422 input tokens and 9,411 output tokens. At standard text rates, the estimate is **$0.02975 for column interpretation only**, excluding the eighteen summaries whose helper does not expose token usage. This is not observed billing and does not account for possible provider cache discounts. Rates: [official GPT-6 Luna model documentation](https://developers.openai.com/api/docs/models/gpt-6-luna).

Reproduce all live timing arms with local fixture files and a configured key (incurs provider usage):

```sh
COLUMN_INTERPRETER_MODEL=gpt-6-luna OPENAI_MODEL=gpt-6-luna \
  backend/venv/bin/python backend/benchmarks/dataset_ai_latency.py \
  --live --output /private/tmp/dataset-ai-results.json
```

Measure only local ingestion without provider calls:

```sh
OPENAI_API_KEY='' COLUMN_INTERPRETER=off OPENAI_MODEL=gpt-6-luna \
  backend/venv/bin/python backend/benchmarks/dataset_ai_latency.py \
  --strategies off --output /private/tmp/dataset-local-results.json
```

Validation: 197 backend tests and 16 interpretation evaluation tests passed with provider access disabled. The new two benchmark tests cover bounded source sampling and the explicit routing behavior. This work adds a measurement harness and report; production ingestion and AI behavior are unchanged.
