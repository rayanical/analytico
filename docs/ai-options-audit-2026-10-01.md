# AI options for Analytico — 2026-10-01

Research checked against opened primary documentation on October 1, 2026. Vendor descriptions establish documented interfaces and vendor claims; this audit did not run comparative model benchmarks or verify account access, SLAs, latency, or calibration on Analytico data. Recommendations below are architectural judgments, not benchmark results.

> Historical research note. Runtime integration descriptions below refer to the pre-fix checkout. Generated Python execution has since been removed, planning is validated, and manual/AI charts share one executor. No new provider has been integrated. See [reliability fixes](local-safety-fixes-2026-10-01.md).

## Recommendation

Fix the boundaries around AI before upgrading the model. Keep numeric parsing, permissions, aggregation, resource limits, and validation in application code. Have AI produce a typed analysis plan using stable column IDs; validate it against the actual dataset and execute approved operations. Generate explanations from computed results with provenance. A better model cannot make unrestricted execution or statistically unjustified imputation reliable.

The existing integration points are `backend/modules/data_janitor.py` (schema interpretation, `gpt-4o-mini`, JSON mode), `backend/services/query_service.py` (query planning, `gpt-4o-mini`, tool arguments/free JSON), and `backend/utils/execution.py` (generated Python execution). This document covers technology choices; the main repo audit should establish specific defects and severity.

## Jev: a useful bounded decision component

TypeSafe's API accepts a state and multiple typed questions. `Choice` selects from supplied options, `Score` evaluates ordered rubric levels, and `Noul` returns a yes/no probability. It can evaluate several questions in one call. These are documented API capabilities, not proof that the selected answers are correct. [TypeSafe quick start](https://docs.typesafe.ai/introduction/quickstart)

**Good pilot candidates (inference):** query intent routing, choosing among candidate columns, classifying semantic column roles, identifying when an ambiguous request needs clarification, and ranking precomputed insight candidates. Supply compact metadata, representative values, and explicit options including “unknown”; use the same ground-truth fixtures as the existing model. Keep authorization entirely in code.

**Poor fit:** numeric value imputation, exact arithmetic, date ordering, arbitrary code generation, or narrative summaries. TypeSafe documents failures in counting, numeric precision, date/time comparisons, indirection, irrelevant long state, adversarial content, and structural invariants. It explicitly recommends code for arithmetic and a generative model for generation. A typed answer can still be the wrong answer. [Jev 1.13 limitations](https://docs.typesafe.ai/model-jaggedness/jev-1.13)

The current documented version is `jev-1.13.0`, with text-only input. The docs specify 64k total context and 32k for state plus the longest question, and explicitly say serving limits may change without notice. They recommend pinning a version when thresholds were tuned against it. TypeSafe says requests/responses are not used for training and enterprise ZDR is available; this is a vendor policy statement that still needs contractual review for sensitive datasets. Avoid depending on `jev-latest` for reproducible behavior. [TypeSafe model reference](https://docs.typesafe.ai/models)

Choice/Score `confidence` is computed from the returned probability distribution; it is not an independently measured correctness probability for Analytico. Noul has no separate confidence property. The vendor itself advises tuning thresholds on domain data. Measure held-out error versus automatic decision coverage, and inspect reliability plots/Brier scores if probability calibration matters. Do not transplant example thresholds into production. [TypeSafe confidence documentation](https://docs.typesafe.ai/confidence)

TypeSafe's launch post presents speed, calibration, and efficiency claims and acknowledges that short dense demo inputs favor its approach. Its workflow evaluations use large external models' reference probabilities. Treat this as vendor evidence, not a guarantee of correctness on uploaded business data. [TypeSafe launch and evaluation explanation](https://typesafe.ai/blog/introducing-system-one-models-and-jev)

## Current OpenAI choices

Official OpenAI documentation currently positions `gpt-6-luna` for focused high-volume work, `gpt-6.1-sol` for balanced intelligence/cost, and `gpt-6-astra` for demanding reasoning/coding. These are sensible candidates for separate routing/extraction, planning, and difficult-query roles, respectively; select using task-level accuracy and end-to-end p95 latency/cost, rather than the vendor hierarchy alone. Keep the current model as a measured baseline. [OpenAI model catalog](https://developers.openai.com/api/docs/models)

A migration requires API/configuration changes, not just replacing the model string. GPT-6 Astra/Sol tool calling requires Responses; reasoning configurations can reject the current sampling parameters. Current guidance also describes caching changes for applications migrating from GPT-5.5 or earlier. Choose one provider adapter with operation-specific configuration and versioned prompts; avoid a large orchestration framework until an actual multi-step workflow justifies it. [Official OpenAI GPT-6 guidance](https://developers.openai.com/api/docs/guides/latest-model)

Use strict Structured Outputs for plans and schema interpretations. Unlike JSON mode, it constrains output to the supported JSON Schema. Handle refusals/incomplete responses and then separately check actual column existence, legal operators, permissions, and sensible semantics. Schema adherence does not establish factual or statistical correctness. Use function calling for tool requests and structured text formats for structured user responses. [OpenAI Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs)

Build portable evaluations in the repo now. The current OpenAI deprecation schedule says Evals becomes read-only October 31, 2026 and its dashboard/API shuts down November 30, 2026. Do not adopt that managed surface as a new foundational dependency. [OpenAI deprecations](https://developers.openai.com/api/docs/deprecations) OpenAI's migration guidance recommends Promptfoo for local/CI evaluations; a small Python harness is also sufficient initially. [OpenAI Evals-to-Promptfoo guide](https://developers.openai.com/cookbook/examples/evaluation/moving-from-openai-evals-to-promptfoo)

## Tabular tools worth considering

| Tool | Documented capability | Practical fit and tradeoff |
| --- | --- | --- |
| scikit-learn imputation | Simple, KNN, and iterative imputers; missing indicators; empty-feature handling. IterativeImputer remains experimental. | Establish measured numerical baselines before a foundation-model imputer. Preserve the missingness mask and distinguish raw from inferred data. Fully empty columns and categorical data need explicit policy. [Imputation guide](https://scikit-learn.org/stable/modules/impute.html) |
| TabPFN | Classification/regression foundation model; missing-value support; GPU recommended. Current README defaults to 3.5. | Explore for a dedicated predictive-analysis feature or benchmarked imputation research, rather than silently rewriting BI data. Current newer model weights are non-commercial even though repository code is Apache 2.0; obtain commercial terms or deliberately evaluate an eligible older version. Check workload size and hardware first. [PriorLabs repository](https://github.com/PriorLabs/TabPFN) |
| Pandera | Dataframe schemas and custom checks; pandas/Polars and other backends. | Useful for explicit ingestion/transformation contracts: dtypes, ranges, uniqueness, and constraints. More valuable for trustworthy data than another AI provider; ordinary assertions are sufficient if the contract remains small. [Pandera repository](https://github.com/unionai-oss/pandera) |
| DuckDB | Analytical execution with configurable resource/security settings. | Potential engine for compiled analysis plans. SQL remains executable code: arbitrary model-generated SQL needs isolation, timeouts, denied external access/extensions, and constrained privileges. Configuration alone is not a sandbox. [DuckDB security guidance](https://duckdb.org/docs/current/operations_manual/securing_duckdb/overview) |

For predictive validation, fit imputers and other preprocessing only on training data and transform validation/test data afterward; use pipelines to preserve this boundary. scikit-learn explicitly includes SimpleImputer in its leakage warnings. This applies when benchmarking a predictive imputer; it does not make imputed values suitable as observed facts in descriptive dashboards. [scikit-learn leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html)

## Adoption order and acceptance evidence

1. Typed plans, deterministic validation/execution, raw-data provenance, and explicit missing-data policy.
2. A representative evaluation set: ambiguous metrics, mixed locales, missing/all-null columns, time boundaries, malicious cell instructions, invalid plans, model refusals/timeouts, and summary claims compared with computed values.
3. Compare the existing model with Luna/Sol on the same fixtures; measure exact result correctness, clarification accuracy, p50/p95 latency, tokens/cost, timeout/fallback rate, and reproducibility across model/prompt versions.
4. Add Jev behind a feature flag for one bounded routing/classification task. Run in shadow mode first; promote only if it improves measured quality or operational cost without increasing incorrect automatic decisions.
5. Explore TabPFN only after defining an explicit predictive feature and checking licensing/data boundaries. Product evidence should precede another infrastructure dependency.

No numeric latency, savings, or calibration benefit is promised here: those need measurements on Analytico's workloads and deployment region.
