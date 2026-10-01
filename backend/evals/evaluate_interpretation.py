#!/usr/bin/env python3
"""Validate interpretation fixtures and score supplied model predictions using stdlib only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
DEFAULT_CASES = HERE / "interpretation_cases.json"
DECISION_FIELDS = (
    "role",
    "unit",
    "parsing_policy",
    "recommended_aggregation",
    "needs_clarification",
)
ACTION_FIELDS = tuple(field for field in DECISION_FIELDS if field != "needs_clarification")
ALLOWED_VALUES = {
    "role": {"identifier", "metric", "temporal", "categorical", "unknown"},
    "unit": {"none", "unknown", "USD", "EUR", "ratio", "mixed_currency", "calendar_date"},
    "parsing_policy": {
        "preserve_lexeme",
        "parse_currency_decimal",
        "retain_currency_identity_unknown",
        "preserve_mixed_currency_values",
        "parse_percent_to_ratio",
        "parse_basis_points_to_ratio",
        "parse_decimal",
        "require_date_locale",
        "preserve_mixed_numeric_formats",
        "preserve_numeric_value",
        "preserve_nulls_parse_numeric",
        "preserve_source",
        "unknown",
    },
    "recommended_aggregation": {
        "count",
        "sum",
        "mean",
        "last_by_entity",
        "ratio_of_sums",
        "none",
        "unknown",
    },
    "needs_clarification": {True, False},
}


class BenchmarkError(ValueError):
    """Raised when a fixture or prediction file does not follow the benchmark contract."""


def _read_json(path: Path) -> Any:
    def reject_nonfinite(value: str) -> None:
        raise BenchmarkError(f"Invalid non-finite JSON number {value!r} in {path}.")

    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle, parse_constant=reject_nonfinite)
    except (OSError, json.JSONDecodeError) as exc:
        raise BenchmarkError(f"Cannot read JSON from {path}: {exc}") from exc


def validate_cases(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise BenchmarkError("Cases must be a schema_version 1 JSON object.")
    if payload.get("decision_fields") != list(DECISION_FIELDS):
        raise BenchmarkError(f"decision_fields must be {list(DECISION_FIELDS)}.")
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise BenchmarkError("cases must be a non-empty list.")

    seen: set[str] = set()
    required_case_keys = {"id", "category", "column_name", "values", "context", "expected"}
    for index, case in enumerate(cases):
        location = f"cases[{index}]"
        if not isinstance(case, dict) or set(case) != required_case_keys:
            raise BenchmarkError(f"{location} must contain exactly {sorted(required_case_keys)}.")
        case_id = case["id"]
        if not isinstance(case_id, str) or not case_id:
            raise BenchmarkError(f"{location}.id must be a non-empty string.")
        if case_id in seen:
            raise BenchmarkError(f"Duplicate case id: {case_id}")
        seen.add(case_id)
        if not all(isinstance(case[key], str) and case[key] for key in ("category", "column_name")):
            raise BenchmarkError(f"{location} category and column_name must be non-empty strings.")
        if not isinstance(case["values"], list):
            raise BenchmarkError(f"{location}.values must be a list; use null for missing values.")
        if not case["values"]:
            raise BenchmarkError(f"{location}.values must include at least one value.")
        if any(not (value is None or isinstance(value, (str, int, float, bool))) for value in case["values"]):
            raise BenchmarkError(f"{location}.values entries must be JSON scalars or null.")
        if not isinstance(case["context"], dict):
            raise BenchmarkError(f"{location}.context must be an object.")
        _validate_decision(case["expected"], location + ".expected", require_complete=True)

    return cases


def _validate_decision(decision: Any, location: str, require_complete: bool = False) -> dict[str, Any]:
    if not isinstance(decision, dict):
        raise BenchmarkError(f"{location} must be an object.")
    unknown_fields = set(decision) - set(DECISION_FIELDS)
    if unknown_fields:
        raise BenchmarkError(f"{location} has unknown decision fields: {sorted(unknown_fields)}")
    if require_complete and set(decision) != set(DECISION_FIELDS):
        raise BenchmarkError(f"{location} must specify all decision fields.")
    for field, value in decision.items():
        if field == "needs_clarification":
            if type(value) is not bool:
                raise BenchmarkError(f"{location}.{field} must be a JSON boolean.")
        elif not isinstance(value, str):
            raise BenchmarkError(f"{location}.{field} must be a string.")
        if value not in ALLOWED_VALUES[field]:
            raise BenchmarkError(f"{location}.{field} has unsupported value {value!r}.")
    return decision


def validate_predictions(payload: Any, case_ids: set[str]) -> tuple[str, dict[str, dict[str, Any]]]:
    if not isinstance(payload, dict) or not isinstance(payload.get("predictions"), list):
        raise BenchmarkError("Predictions must be an object containing a predictions list.")
    model = payload.get("model", "unnamed")
    if not isinstance(model, str) or not model:
        raise BenchmarkError("model must be a non-empty string when supplied.")

    by_id: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(payload["predictions"]):
        location = f"predictions[{index}]"
        if not isinstance(row, dict) or not {"case_id", "decision"}.issubset(row):
            raise BenchmarkError(f"{location} must include case_id and decision.")
        if set(row) - {"case_id", "decision", "observed"}:
            raise BenchmarkError(f"{location} contains unsupported top-level fields.")
        case_id = row["case_id"]
        if not isinstance(case_id, str):
            raise BenchmarkError(f"{location}.case_id must be a string.")
        if case_id not in case_ids:
            raise BenchmarkError(f"{location} references unknown case id {case_id!r}.")
        if case_id in by_id:
            raise BenchmarkError(f"Duplicate prediction for case id {case_id!r}.")
        decision = _validate_decision(row["decision"], location + ".decision")
        observed = row.get("observed", {})
        if not isinstance(observed, dict):
            raise BenchmarkError(f"{location}.observed must be an object when supplied.")
        by_id[case_id] = {"decision": decision, "observed": observed}
    return model, by_id


def project_candidate_input(case: dict[str, Any]) -> dict[str, Any]:
    """Return external routing metadata and the only fields candidate adapters may consume."""
    return {
        "case_id": case["id"],
        "input": {
            "column_name": case["column_name"],
            "values": case["values"],
            "context": case["context"],
        },
    }


def score(cases: list[dict[str, Any]], model: str, predictions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    total = len(cases)
    supplied = len(predictions)
    field_metrics: dict[str, dict[str, Any]] = {}
    for field in DECISION_FIELDS:
        available = [case for case in cases if field in predictions.get(case["id"], {}).get("decision", {})]
        correct = sum(
            predictions[case["id"]]["decision"][field] == case["expected"][field]
            for case in available
        )
        field_metrics[field] = {
            "scored": len(available),
            "coverage": _ratio(len(available), total),
            "correct": correct,
            "accuracy": _ratio(correct, len(available)),
        }

    complete = [
        case
        for case in cases
        if set(predictions.get(case["id"], {}).get("decision", {})) == set(DECISION_FIELDS)
    ]
    exact = sum(
        all(predictions[case["id"]]["decision"][field] == case["expected"][field] for field in DECISION_FIELDS)
        for case in complete
    )

    clarification_rows = [
        case for case in cases
        if "needs_clarification" in predictions.get(case["id"], {}).get("decision", {})
    ]
    true_positive = sum(
        predictions[case["id"]]["decision"]["needs_clarification"] is True
        and case["expected"]["needs_clarification"] is True
        for case in clarification_rows
    )
    false_positive = sum(
        predictions[case["id"]]["decision"]["needs_clarification"] is True
        and case["expected"]["needs_clarification"] is False
        for case in clarification_rows
    )
    false_negative = sum(
        predictions[case["id"]]["decision"]["needs_clarification"] is False
        and case["expected"]["needs_clarification"] is True
        for case in clarification_rows
    )
    true_negative = sum(
        predictions[case["id"]]["decision"]["needs_clarification"] is False
        and case["expected"]["needs_clarification"] is False
        for case in clarification_rows
    )

    auto_rows = [
        case for case in cases
        if predictions.get(case["id"], {}).get("decision", {}).get("needs_clarification") is False
    ]
    wrong_auto = 0
    for case in auto_rows:
        decision = predictions[case["id"]]["decision"]
        if case["expected"]["needs_clarification"] is True:
            wrong_auto += 1
        elif any(
            field not in decision or decision[field] != case["expected"][field]
            for field in ACTION_FIELDS
        ):
            wrong_auto += 1

    gold_clarification = sum(case["expected"]["needs_clarification"] for case in cases)
    scored_gold_clarification = sum(
        case["expected"]["needs_clarification"] for case in clarification_rows
    )
    scored_gold_no_clarification = len(clarification_rows) - scored_gold_clarification
    unnecessary_clarifications = sum(
        predictions[case["id"]]["decision"].get("needs_clarification") is True
        and case["expected"]["needs_clarification"] is False
        for case in cases
        if case["id"] in predictions
    )

    return {
        "model": model,
        "cases": total,
        "prediction_coverage": _ratio(supplied, total),
        "predicted_cases": supplied,
        "complete_decision_coverage": _ratio(len(complete), total),
        "complete_decisions": len(complete),
        "exact_decisions": exact,
        "exact_decision_accuracy": _ratio(exact, len(complete)),
        "per_field": field_metrics,
        "clarification": {
            "scored": len(clarification_rows),
            "accuracy": _ratio(true_positive + true_negative, len(clarification_rows)),
            "precision": _ratio(true_positive, true_positive + false_positive),
            "recall": _ratio(true_positive, true_positive + false_negative),
            "true_positive": true_positive,
            "false_positive_unnecessary": false_positive,
            "false_negative_missed": false_negative,
            "gold_clarification_cases": gold_clarification,
            "scored_gold_clarification_cases": scored_gold_clarification,
            "missed_clarification_rate": _ratio(false_negative, scored_gold_clarification),
            "scored_gold_no_clarification_cases": scored_gold_no_clarification,
            "unnecessary_clarification_rate": _ratio(unnecessary_clarifications, scored_gold_no_clarification),
        },
        "wrong_auto_decisions": {
            "count": wrong_auto,
            "auto_decisions": len(auto_rows),
            "rate": _ratio(wrong_auto, len(auto_rows)),
            "rule": "An auto decision is wrong if clarification was required, an action field is missing, or any supplied action field differs from expected.",
        },
    }


def _ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES, help="fixture JSON path")
    parser.add_argument("--predictions", type=Path, help="prediction JSON path; omissions are reported as uncovered")
    parser.add_argument("--export-candidate-inputs", type=Path, help="write leak-safe model input envelopes and exit")
    parser.add_argument("--model-name", help="override the model label in the report")
    parser.add_argument("--self-check", action="store_true", help="validate the fixture file and exit")
    parser.add_argument("--write-report", type=Path, help="also write the report JSON to this path")
    args = parser.parse_args(argv)

    try:
        cases = validate_cases(_read_json(args.cases))
        if args.export_candidate_inputs:
            exported = {
                "schema_version": 1,
                "inputs": [project_candidate_input(case) for case in cases],
            }
            rendered = json.dumps(exported, indent=2, sort_keys=True) + "\n"
            args.export_candidate_inputs.parent.mkdir(parents=True, exist_ok=True)
            args.export_candidate_inputs.write_text(rendered, encoding="utf-8")
            sys.stdout.write(f"Wrote {len(cases)} leak-safe candidate inputs to {args.export_candidate_inputs}\n")
            return 0
        if args.self_check:
            result: Any = {"schema": "ok", "cases": len(cases), "decision_fields": list(DECISION_FIELDS)}
        else:
            if args.predictions is None:
                parser.error("--predictions is required unless --self-check is used")
            payload = _read_json(args.predictions)
            model, predictions = validate_predictions(payload, {case["id"] for case in cases})
            result = score(cases, args.model_name or model, predictions)
        rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
        if args.write_report:
            args.write_report.parent.mkdir(parents=True, exist_ok=True)
            args.write_report.write_text(rendered, encoding="utf-8")
        sys.stdout.write(rendered)
        return 0
    except BenchmarkError as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
