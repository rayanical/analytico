# Interpretation benchmark

This is a small, offline benchmark for comparing column interpretation decisions before evaluating Jev or Luna. It uses synthetic values and reviewed expected decisions; it makes no provider calls and does not implement either model.

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

## Current-repo baseline

`backend/evals/baseline_current_repo.json` is a point-in-time offline snapshot from the current ingestion and profiling path. Regenerate it with `snapshot_current_repo.py` in an environment with the backend dependencies installed; the command above uses this checkout's local environment. The adapter feeds only each case's column name and values to `clean_dataframe`, `detect_semantic_type`, and `auto_profile`; it never reads expected decisions. It forces `OPENAI_API_KEY` empty before importing backend code, which skips optional schema enrichment before it can construct a client. No provider is called. The snapshot records Python and pandas versions and marks its source as the working tree, without claiming a commit revision. It records the runtime role and, only when emitted by auto-profile, its default aggregation. `observed` retains the runtime column format and cleaning actions for context.

Current code does not emit contract-level units, explicit parsing policies, or clarification decisions. Those fields are omitted and shown as zero coverage by the scorer; they are not filled with guessed defaults. Existing format labels such as `currency` and `percentage` are not currency identities or explicit unit decisions.

The measured snapshot covers all 15 case IDs for role (7/15 correct, 46.7%). It emitted an auto-profile aggregation for 6/15 cases (40% coverage); 2/6 matched the benchmark recommendation (33.3%). It classified all-missing and ambiguous date fields as categorical, treated the `account_balance_usd` header as an identifier, and chose an unweighted mean for both rate cases, including the one without a denominator. The explicit USD fixture remained text and was labeled categorical despite a generic `currency` format because the current parser could not safely parse the mix of grouped and ungrouped values. The dollar-only case was parsed and summed even though `$` does not establish a currency code. These examples describe the curated synthetic fixture, not a population estimate.

The baseline adapter snapshot uses the actual current backend functions rather than a reimplementation of their rules. The scoring/evaluation script and its tests use only the Python standard library. The checked-in baseline is intentionally static so later runtime changes do not silently rewrite the comparison point.
