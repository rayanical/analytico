# Interpretation benchmark

This benchmark compares column interpretation decisions using synthetic values and reviewed expected decisions. Validation and scoring are offline; a separate opt-in runner calls Jev or Luna.

The fixtures cover identifiers, explicit and ambiguous currencies, mixed currencies, percentages and basis points, balance stocks, ambiguous dates and number formats, missing values, unknown units, and rates that need weights or denominators. Each case keeps its expected decision separate from model predictions in `backend/evals/interpretation_cases.json`.

The decision contract has five fields: `role`, `unit`, `parsing_policy`, `recommended_aggregation`, and `needs_clarification`. The JSON fixture defines the allowed value choices. Use `unknown` when evidence is insufficient and `none` when the concept does not apply. An explicit `unknown` is still a prediction and is scored against expected truth; it is not an abstention. Omitted fields are unmeasured and reduce field coverage. The benchmark does not accept confidence scores and makes no calibration claim.

Candidate adapters should call `project_candidate_input` or use the export command below. It keeps `case_id` as routing metadata and places only `column_name`, `values`, and `context` under `input`. Pass only that `input` object to a candidate. Fixture category labels, case IDs, and expected decisions must stay outside the model request.

Run the standard-library schema check and fixture tests, regenerate the offline current-repo snapshot with the backend environment, then score it with the standard-library evaluator:

```sh
python backend/evals/evaluate_interpretation.py --self-check
python -m unittest discover -s backend/evals -p 'test_*.py' -v
backend/venv/bin/python backend/evals/snapshot_current_repo.py
python backend/evals/evaluate_interpretation.py --predictions backend/evals/baseline_current_repo.json
python backend/evals/evaluate_interpretation.py --export-candidate-inputs /tmp/interpretation-candidates.json
```

To score Jev, Luna, or another candidate, provide JSON with one row per predicted case. Each `decision` can be partial during early adapter work, but full decisions are needed for exact-decision scoring. Unknown case IDs, duplicate predictions, unsupported values, and malformed field types are rejected.

```json
{
  "model": "Jev",
  "predictions": [
    {
      "case_id": "currency.usd_explicit",
      "decision": {
        "role": "metric",
        "unit": "USD",
        "parsing_policy": "parse_currency_decimal",
        "recommended_aggregation": "sum",
        "needs_clarification": false
      }
    }
  ]
}
```

```sh
python backend/evals/evaluate_interpretation.py \
  --predictions /path/to/jev-predictions.json \
  --write-report /tmp/jev-interpretation-report.json
```

The report includes prediction and complete-decision coverage, exact decisions, per-field scored coverage and accuracy, clarification precision/recall, and wrong auto decisions. A wrong auto decision includes proceeding when the expected decision requests clarification, omitting an action field, or choosing any incorrect action field. Clarification rates use only cases where the prediction supplied that field; unsupported runtime behavior is not recorded as a false or correct clarification decision.

The mixed-number fixture intentionally contains three conventions: comma grouping with a decimal point, dot grouping with a decimal comma, and space grouping with a decimal comma. Its expected policy is to preserve the values and ask how to interpret the mixed conventions; one locale setting cannot resolve that fixture.

## Opt-in Jev and Luna run

`backend/evals/run_interpretation.py` exports the same leak-safe candidate inputs by default and makes no network calls. To run the live comparison, pass both `--live` and a provider (`jev` or `luna`) and provide an output path. The runner loads `backend/.env` without overriding variables already exported by the shell. Jev requires `AI_GATEWAY_API_KEY`; Luna requires `OPENAI_API_KEY`. It checks the selected key before importing or calling the provider adapter.

```sh
# Offline preview; safe to run without provider credentials.
backend/venv/bin/python backend/evals/run_interpretation.py \
  --output /tmp/interpretation-inputs.json

# Explicit live run. Results are saved even when one or more calls fail.
backend/venv/bin/python backend/evals/run_interpretation.py \
  --live --provider jev \
  --output /tmp/jev-live-interpretation.json

backend/venv/bin/python backend/evals/run_interpretation.py \
  --live --provider luna \
  --output /tmp/luna-live-interpretation.json
```

The adapter receives only `column_name`, `values`, and `context`; `case_id` remains local routing metadata. Dry-run exports also omit fixture categories and expected decisions. Live output keeps decisions separate from provider metadata, and feeds the decisions as returned into the existing `score` function. It does not rewrite recommendations to fit runtime executor capabilities, so semantic interpretation accuracy and runtime support remain separate concerns. Failed calls produce rows with sanitized error codes, preserve any completed predictions, save the report, and return a nonzero exit status. A missing key produces unavailable rows without making any provider calls.

Latency summaries report p50 and p95 with the number of measured samples and use linear interpolation over sorted samples (type 7). Token totals include only usage values actually returned by the adapter; each total shows how many cases reported it. Cost is `unknown` unless the adapter reports a cost for every case with a consistent currency. No live provider calls are part of the offline fixture or runner tests.

## Current-repo baseline

`backend/evals/baseline_current_repo.json` is a point-in-time offline snapshot from the current ingestion and profiling path. Regenerate it with `snapshot_current_repo.py` in an environment with the backend dependencies installed; the command above uses this checkout's local environment. The adapter feeds only each case's column name and values to `clean_dataframe`, `detect_semantic_type`, and `auto_profile`; it never reads expected decisions. It forces `OPENAI_API_KEY` empty and `COLUMN_INTERPRETER=off` before importing backend code, so it cannot construct an interpretation provider client. No provider is called. The snapshot records Python and pandas versions and marks its source as the working tree, without claiming a commit revision. It records the runtime role and, only when emitted by auto-profile, its default aggregation. `observed` retains the runtime column format and cleaning actions for context.

The offline baseline does not emit contract-level units, explicit parsing policies, or clarification decisions. Those fields are omitted and shown as zero coverage by the scorer; they are not filled with guessed defaults. Existing format labels such as `currency` and `percentage` are not currency identities or explicit unit decisions.

The measured snapshot covers all 15 case IDs for role (7/15 correct, 46.7%). It emitted an auto-profile aggregation for 6/15 cases (40% coverage); 2/6 matched the benchmark recommendation (33.3%). It classified all-missing and ambiguous date fields as categorical, treated the `account_balance_usd` header as an identifier, and chose an unweighted mean for both rate cases, including the one without a denominator. The explicit USD fixture remained text and was labeled categorical despite a generic `currency` format because the current parser could not safely parse the mix of grouped and ungrouped values. The dollar-only case was parsed and summed even though `$` does not establish a currency code. These examples describe the curated synthetic fixture, not a population estimate.

The baseline adapter snapshot uses the actual current backend functions rather than a reimplementation of their rules. The scoring/evaluation script and its tests use only the Python standard library. The checked-in baseline is intentionally static so later runtime changes do not silently rewrite the comparison point.

### First live Luna run (October 1, 2026, local time)

[Recorded report](../backend/evals/results/luna-2026-10-01.json), prompt `column-interpretation-v1`, resolved model `gpt-6-luna`, fifteen synthetic cases, one request per case:

- All fifteen requests returned complete decisions without provider failures.
- Role accuracy: 15/15. Complete five-field match: 9/15. Clarification accuracy: 14/15, including one missed clarification among seven cases requiring it.
- The scorer flags two of nine automatic decisions as wrong. One guessed USD from a bare dollar symbol; runtime source-evidence checks reject that conversion. The other selected a currency parsing policy instead of the fixture's decimal parsing policy for an explicit USD balance.
- Latency: p50 1,213.9 ms, p95 2,117.01 ms, measured end to end over fifteen requests.
- Reported usage: 17,821 input and 623 output tokens. Actual billed cost was not returned and remains unknown.

This single small run does not establish production accuracy or a Jev/Luna winner. Jev live testing remains pending a local `AI_GATEWAY_API_KEY`; mocked contract validation does not measure model quality. Interpretation stays off by default. A wider fixture set and an interpretation review UI should precede default-on ingestion.
