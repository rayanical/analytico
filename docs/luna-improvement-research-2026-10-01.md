# Improving Luna column interpretation

Research snapshot: October 1, 2026. This note uses official OpenAI documentation and the current adapter and benchmark. No provider/API inference calls were made.

## Setup at the start of this iteration

`backend/modules/column_interpretation.py` defaults the Luna provider to `gpt-6-luna` and sends a Responses API request with strict Structured Outputs, `reasoning.effort="none"`, `store=false`, and a 300-token output cap. `COLUMN_INTERPRETER_MODEL` can override the default, so the code targets Luna by default but a deployed configuration can select another model. OpenAI lists Luna as suited to focused, high-volume work, supports Structured Outputs, and supports both `none` and `low` reasoning effort. [GPT-6 Luna model](https://developers.openai.com/api/docs/models/gpt-6-luna)

The request already has useful guardrails: bounded JSON input, an explicit untrusted-data instruction, named allowed choices with meanings, an `unknown` option, and a `needs_clarification` field. It repeats some of the same rules in the system and user messages. The strict schema constrains the five fields and their enums; the response parser separately handles refusals, incomplete output, and cross-field checks.

The existing interpretation benchmark has 15 synthetic cases. Its first live Luna run made one request per case and got 9/15 complete five-field decisions exactly right, with one missed clarification and two decisions the scorer flags as unsafe to automate. This is a useful failure-finding baseline, not evidence of production reliability or coverage of all edge cases. See [the benchmark](interpretation-benchmark.md).

## Prompt changes worth evaluating

OpenAI recommends straightforward, direct prompts for reasoning models, avoiding “think step by step” requests, and trying zero-shot before few-shot. It recommends examples only when task requirements need them, with examples that closely match the instructions; delimiters can make input sections clearer. [Reasoning best practices](https://developers.openai.com/api/docs/guides/reasoning-best-practices)

For this adapter, first test a shorter single source of task rules. Keep the policy that evidence must come only from the column name, samples, and profile; treat those values as untrusted data; choose `unknown` and request clarification when a choice is unsupported; never infer currency identity, date order, measurement grain, denominators, or aggregation semantics from weak clues. Put the allowed-value definitions and the state in clearly named, delimited sections. Removing duplicated wording may lower token use and reduce rule drift, but retain any wording that the eval shows protects against an unsafe guess.

Add examples only for mistakes that remain after the zero-shot prompt has been measured. A small contrastive set should target real boundaries in this task, such as an explicit `USD` code versus a bare `$`, a time-varying balance snapshot versus additive transactions, an ISO date versus an ambiguous numeric date, and a homogeneous numeric format versus mixed locale conventions. Each example should make the evidence difference visible and show `unknown`/clarification when the distinction cannot be established. Do not put eval labels or case IDs into the prompt, and keep examples out of the held-out set. The tradeoff is that a few examples can teach distinctions, while too many or overly similar examples can bias the model toward the examples instead of the general rule.

OpenAI also recommends clear schema keys and descriptions for important fields, then using evals to determine whether the schema helps generation. Add concise field descriptions to the schema if the current enum names and prompt meanings leave repeated field-level errors; measure that change separately from prompt changes. [Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs)

## `none` versus `low`

OpenAI describes `none` as appropriate for latency-critical tasks that do not need reasoning, and specifically lists classification as a common use. `low` adds modest latency and is intended for efficient reasoning in more involved tasks. Lower effort generally favors speed and fewer reasoning tokens; higher effort can improve quality when more complete reasoning helps. [Reasoning guide](https://developers.openai.com/api/docs/guides/reasoning)

| Setting | Fit for this classifier | Tradeoff to measure |
|---|---|---|
| `none` | Strong starting point for a bounded, single-turn classification request; it is the current setting. | Lowest latency and token overhead, with possible weakness on ambiguous inputs and interactions among the five fields. |
| `low` | A plausible candidate because unit, parsing, aggregation, and clarification can depend on the same sparse evidence. | May reduce unsafe guesses, but adds latency and reasoning-token cost; OpenAI does not promise a quality gain for this particular task. |

Run a paired comparison on the same inputs with the same prompt, model ID, schema, and output cap; change only effort. Repeat cases to measure how often each setting changes its decision. Keep `none` if `low` does not reduce unsafe automatic decisions enough to justify its latency and cost. Consider `low` only if the held-out results show a meaningful improvement. This is a task-specific experiment, not a claim that one setting is universally safer.

## What Structured Outputs does and does not guarantee

Strict Structured Outputs is the right format for this response contract: it constrains the response to the declared shape and allowed values. Luna supports it. The feature does not establish that a semantically valid enum is correct. OpenAI explicitly warns that structured responses can still contain mistakes, and that an unrelated input may be forced into the requested schema and produce hallucinated field values. [Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs)

Keep an explicit abstention path in the schema and prompt. Then keep semantic and source-evidence checks in deterministic application code: for example, a bare currency symbol must not become `USD`; ambiguous dates must not be assigned a locale; and unsupported weighted-rate or snapshot aggregation must not be silently treated as an ordinary mean or sum. JSON Schema also does not support conditional keywords such as `if`/`then`/`else` in Structured Outputs, so relationships among fields belong in the application validator. The current parser enforces some consistency rules, but successful schema parsing alone must never authorize parsing or aggregation.

Continue detecting refusal and incomplete responses before parsing, and retain the fail-closed result path on malformed or contradictory outputs. Structured Outputs documentation calls out refusals and token-limited incomplete responses as cases applications must handle. [Structured Outputs edge cases](https://developers.openai.com/api/docs/guides/structured-outputs)

## A repeatable improvement loop

OpenAI says model output is nondeterministic and can change between snapshots and model families. Its optimization guidance calls for representative eval data, a baseline, prompt changes based on results, and repeating the loop. Its eval guidance recommends typical, edge, and adversarial examples with expert labels; continuous evaluation should grow as new nondeterministic failures are found. [Model optimization](https://developers.openai.com/api/docs/guides/model-optimization), [evaluation best practices](https://developers.openai.com/api/docs/guides/evaluation-best-practices)

For Luna, use this loop:

1. Preserve the current 15-case run as the baseline. Expand the labeled set with reviewed, production-like columns and deliberately difficult cases: mixed currencies and numeric locales, ambiguous and missing values, identifiers that resemble measures, rates without denominators, snapshots without grain/time context, misleading headers, hostile text inside sample values, and unrelated or empty states.
2. Split examples by semantic pattern, not just random row, into a prompt-tuning set and a locked holdout. Keep the benchmark's expected answers and routing metadata out of the model input. Label cases with a domain reviewer before judging outputs.
3. Compare one change at a time: concise zero-shot prompt, targeted examples if needed, then `none` versus `low`. Repeat the same cases under each candidate and report decision agreement across repeats as well as average accuracy.
4. Make unsafe automation the primary failure metric: wrong automatic decisions and missed clarifications. Also report exact five-field accuracy, per-field accuracy and coverage, clarification precision/recall, refusals/incomplete/failures, and p50/p95 latency and token use. OpenAI recommends classification/scoring-style evals and calibration against human assessment rather than relying on a subjective impression that results look good.
5. Promote a candidate only when it improves the predeclared safety metric on the tuning set and holds on the untouched test set without a critical regression. Add newly discovered failures to future regression data, record prompt version, requested/resolved model and effort, and keep checking after updates.

There is no finite test set or number of repeated calls that proves safety against every possible column name, data shape, or adversarial value. A defensible claim is bounded: the model and prompt meet named metrics on reviewed datasets, with deterministic validation and clarification keeping uncertain cases from being applied automatically. The current 15-case run does not meet a basis for claiming broad safety; it does identify concrete prompt and eval improvements to measure.
