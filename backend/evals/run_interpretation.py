#!/usr/bin/env python3
"""Export safe benchmark inputs or opt in to live Jev/Luna interpretation calls."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
from functools import partial
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from evaluate_interpretation import (
    BenchmarkError,
    DEFAULT_CASES,
    _read_json,
    project_candidate_input,
    score,
    validate_cases,
    validate_predictions,
)


HERE = Path(__file__).resolve().parent
BACKEND_ROOT = HERE.parent
PROTECTED_OUTPUTS = {DEFAULT_CASES.resolve(), (HERE / "baseline_current_repo.json").resolve()}
ALLOWED_STATUSES = {"ok", "disabled", "unavailable", "error", "uncertain"}
FAILURE_STATUSES = {"disabled", "unavailable", "error"}
PROVIDER_KEYS = {"jev": "AI_GATEWAY_API_KEY", "luna": "OPENAI_API_KEY"}
KNOWN_ERROR_CODES = {
    "invalid_input",
    "invalid_provider",
    "invalid_model",
    "invalid_reasoning_effort",
    "missing_api_key",
    "request_timeout",
    "provider_unavailable",
    "provider_http_error",
    "provider_refusal",
    "incomplete_response",
    "invalid_response",
    "disabled",
    "provider_error",
    "adapter_unavailable",
    "invalid_decision",
}
TOKEN_FIELDS = ("input_tokens", "output_tokens", "total_tokens")


def _load_backend_environment() -> None:
    """Load backend/.env without overriding environment variables set by the caller."""
    from dotenv import load_dotenv

    load_dotenv(dotenv_path=BACKEND_ROOT / ".env", override=False)


def _is_configured_key(value: str | None) -> bool:
    if not value or not value.strip():
        return False
    normalized = value.strip().lower()
    placeholders = ("your-", "your_", "replace-", "placeholder")
    return normalized != "changeme" and not normalized.startswith(placeholders)


def _provider_key_available(provider: str) -> bool:
    return _is_configured_key(os.environ.get(PROVIDER_KEYS[provider]))


def _load_interpret_column() -> Callable[..., Any]:
    if str(BACKEND_ROOT) not in sys.path:
        sys.path.insert(0, str(BACKEND_ROOT))
    from modules.column_interpretation import interpret_column

    return interpret_column


def _decision_payload(decision: Any) -> dict[str, Any] | None:
    if decision is None:
        return None
    if isinstance(decision, dict):
        return decision
    model_dump = getattr(decision, "model_dump", None)
    if callable(model_dump):
        try:
            return model_dump(mode="json")
        except TypeError:
            return model_dump()
    raise TypeError("decision is not a serializable interpretation decision")


def _clean_error_code(value: Any, fallback: str = "provider_error") -> str:
    if isinstance(value, str) and value in KNOWN_ERROR_CODES:
        return value
    return fallback


def _clean_usage(value: Any) -> dict[str, Any]:
    """Keep provider usage data JSON-safe and bounded, without inventing token counts."""
    if not isinstance(value, dict):
        return {}
    cleaned: dict[str, Any] = {}
    for key in TOKEN_FIELDS:
        count = value.get(key)
        if type(count) is int and count >= 0:
            cleaned[key] = count
    # Preserve a directly reported cost only when the adapter supplies a finite number.
    cost = value.get("cost")
    if isinstance(cost, (int, float)) and not isinstance(cost, bool) and math.isfinite(cost) and cost >= 0:
        cleaned["cost"] = cost
        currency = value.get("cost_currency")
        if isinstance(currency, str) and len(currency) <= 12:
            cleaned["cost_currency"] = currency
    return cleaned


def _result_row(
    case_id: str,
    provider: str,
    *,
    status: str,
    resolved_provider: Any = None,
    decision: dict[str, Any] | None = None,
    model: Any = None,
    prompt_version: Any = None,
    reasoning_effort: Any = None,
    latency_ms: Any = None,
    usage: Any = None,
    confidence: Any = None,
    error_code: Any = None,
) -> dict[str, Any]:
    safe_status = status if status in ALLOWED_STATUSES else "error"
    if error_code is None and safe_status in FAILURE_STATUSES:
        error_code = {
            "disabled": "disabled",
            "unavailable": "provider_unavailable",
            "error": "provider_error",
        }[safe_status]
    safe_latency = latency_ms
    if not isinstance(safe_latency, (int, float)) or isinstance(safe_latency, bool) or not math.isfinite(safe_latency) or safe_latency < 0:
        safe_latency = None
    safe_confidence = confidence if isinstance(confidence, (dict, int, float, str, bool)) else None
    if isinstance(safe_confidence, float) and not math.isfinite(safe_confidence):
        safe_confidence = None
    if isinstance(safe_confidence, dict):
        try:
            json.dumps(safe_confidence, allow_nan=False)
        except (TypeError, ValueError):
            safe_confidence = None
    return {
        "case_id": case_id,
        "status": safe_status,
        "decision": decision,
        "provider_metadata": {
            "provider": resolved_provider if isinstance(resolved_provider, str) and resolved_provider else provider,
            "model": model if isinstance(model, str) and model else None,
            "prompt_version": prompt_version if isinstance(prompt_version, str) and prompt_version else None,
            "reasoning_effort": reasoning_effort if reasoning_effort in {"none", "low"} else None,
            "latency_ms": safe_latency,
            "usage": _clean_usage(usage),
            "confidence": safe_confidence,
            "error_code": _clean_error_code(error_code) if error_code is not None else None,
        },
    }


def _failed_row(case_id: str, provider: str, code: str, *, status: str = "unavailable") -> dict[str, Any]:
    return _result_row(case_id, provider, status=status, error_code=code)


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return round(ordered[lower], 3)
    fraction = position - lower
    return round(ordered[lower] + (ordered[upper] - ordered[lower]) * fraction, 3)


def _latency_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = [
        row["provider_metadata"]["latency_ms"]
        for row in rows
        if row["provider_metadata"]["latency_ms"] is not None
    ]
    return {
        "samples": len(values),
        "p50_ms": _quantile(values, 0.50),
        "p95_ms": _quantile(values, 0.95),
        "quantile_method": "linear interpolation over sorted samples (type 7)",
    }


def _usage_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total_cases = len(rows)
    token_totals: dict[str, Any] = {}
    for field in ("input_tokens", "output_tokens"):
        observed = [
            row["provider_metadata"]["usage"][field]
            for row in rows
            if field in row["provider_metadata"]["usage"]
        ]
        token_totals[field] = {
            "total": sum(observed) if observed else None,
            "reported_cases": len(observed),
            "total_cases": total_cases,
        }

    reported_totals = [
        row["provider_metadata"]["usage"]["total_tokens"]
        for row in rows
        if "total_tokens" in row["provider_metadata"]["usage"]
    ]
    derived_totals = [
        usage["input_tokens"] + usage["output_tokens"]
        for row in rows
        if "total_tokens" not in (usage := row["provider_metadata"]["usage"])
        and "input_tokens" in usage
        and "output_tokens" in usage
    ]
    token_totals["total_tokens"] = {
        "total": sum(reported_totals) + sum(derived_totals) if reported_totals or derived_totals else None,
        "reported_cases": len(reported_totals),
        "derived_cases": len(derived_totals),
        "total_cases": total_cases,
    }

    cost_rows = [row["provider_metadata"]["usage"] for row in rows if "cost" in row["provider_metadata"]["usage"]]
    currencies = {usage.get("cost_currency") for usage in cost_rows}
    cost_known_for_all = (
        len(cost_rows) == total_cases
        and len(currencies) == 1
        and isinstance(next(iter(currencies)), str)
        and bool(next(iter(currencies)))
    )
    cost = {
        "total": sum(usage["cost"] for usage in cost_rows) if cost_known_for_all else None,
        "currency": next(iter(currencies)) if cost_known_for_all else None,
        "reported_cases": len(cost_rows),
        "total_cases": total_cases,
        "status": "reported" if cost_known_for_all else "unknown",
    }
    return {"tokens": token_totals, "cost": cost}


def _validate_decision(case_id: str, decision: dict[str, Any], case_ids: set[str]) -> bool:
    try:
        validate_predictions(
            {"predictions": [{"case_id": case_id, "decision": decision}]},
            case_ids,
        )
    except BenchmarkError:
        return False
    return True


def run_live(
    cases: list[dict[str, Any]],
    provider: str,
    interpret: Callable[..., Any] | None = None,
    *,
    api_key_available: bool = True,
    reasoning_effort: str = "none",
) -> dict[str, Any]:
    """Run the adapter on leak-safe inputs; the callable seam keeps tests offline."""
    started_at = datetime.now(timezone.utc).isoformat()
    case_ids = {case["id"] for case in cases}
    rows: list[dict[str, Any]] = []

    # This guard runs before adapter import and before any provider callable can be invoked.
    if not api_key_available:
        rows = [_failed_row(case["id"], provider, "missing_api_key") for case in cases]
    else:
        if interpret is None:
            try:
                interpret = _load_interpret_column()
                if provider == "luna":
                    interpret = partial(interpret, reasoning_effort=reasoning_effort)
            except Exception:
                rows = [_failed_row(case["id"], provider, "adapter_unavailable") for case in cases]

        if interpret is not None:
            for case in cases:
                projected = project_candidate_input(case)
                case_id = projected["case_id"]
                only_model_input = copy.deepcopy(projected["input"])
                try:
                    result = interpret(only_model_input, provider=provider)
                    status = getattr(result, "status", "error")
                    decision = _decision_payload(getattr(result, "decision", None))
                    error_code = getattr(result, "error_code", None)
                    if status not in ALLOWED_STATUSES:
                        status = "error"
                        error_code = "invalid_response"
                    if decision is not None and not _validate_decision(case_id, decision, case_ids):
                        error_code = "invalid_decision"
                    elif status == "ok" and decision is None:
                        status = "error"
                        error_code = "invalid_response"
                    row = _result_row(
                        case_id,
                        provider,
                        status=status,
                        resolved_provider=getattr(result, "provider", None),
                        decision=decision,
                        model=getattr(result, "model", None),
                        prompt_version=getattr(result, "prompt_version", None),
                        reasoning_effort=getattr(result, "reasoning_effort", None),
                        latency_ms=getattr(result, "latency_ms", None),
                        usage=getattr(result, "usage", None),
                        confidence=getattr(result, "confidence", None),
                        error_code=error_code,
                    )
                    if decision is not None and not _validate_decision(case_id, decision, case_ids):
                        row["decision"] = decision
                        row["status"] = "error"
                        row["provider_metadata"]["error_code"] = "invalid_decision"
                    rows.append(row)
                except Exception:
                    rows.append(_failed_row(case_id, provider, "provider_error", status="error"))

    prediction_rows = [
        {"case_id": row["case_id"], "decision": row["decision"]}
        for row in rows
        if row["decision"] is not None
        and _validate_decision(row["case_id"], row["decision"], case_ids)
    ]
    _, predictions = validate_predictions(
        {"model": "provider_predictions", "predictions": prediction_rows},
        case_ids,
    )
    scored = score(cases, "provider_predictions", predictions)
    scored.pop("model", None)

    provider_models = sorted(
        {
            row["provider_metadata"]["model"]
            for row in rows
            if row["provider_metadata"]["model"] is not None
        }
    )
    resolved_providers = sorted(
        {
            row["provider_metadata"]["provider"]
            for row in rows
            if row["provider_metadata"]["provider"] is not None
        }
    )
    prompt_versions = sorted(
        {
            row["provider_metadata"]["prompt_version"]
            for row in rows
            if row["provider_metadata"]["prompt_version"] is not None
        }
    )
    failure_count = sum(row["status"] in FAILURE_STATUSES for row in rows)
    return {
        "schema_version": 1,
        "mode": "live",
        "provider_run": {
            "requested_provider": provider,
            "requested_reasoning_effort": reasoning_effort if provider == "luna" else None,
            "resolved_providers": resolved_providers,
            "models": provider_models,
            "prompt_versions": prompt_versions,
            "started_at_utc": started_at,
        },
        "results": rows,
        "scoring": scored,
        "metrics": {
            "latency_ms": _latency_summary(rows),
            "usage": _usage_summary(rows),
        },
        "failure_count": failure_count,
    }


def dry_run_report(cases: list[dict[str, Any]], provider: str | None) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "mode": "dry_run",
        "provider_requested": provider,
        "inputs": [project_candidate_input(case) for case in cases],
    }


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES, help="fixture JSON path")
    parser.add_argument("--provider", choices=tuple(PROVIDER_KEYS), help="provider used in live mode")
    parser.add_argument("--reasoning-effort", choices=("none", "low"), default="none", help="Luna reasoning experiment; runtime defaults to none")
    parser.add_argument("--live", action="store_true", help="make provider calls; omitted by default")
    parser.add_argument("--output", type=Path, help="write run report JSON to this path")
    args = parser.parse_args(argv)

    try:
        cases = validate_cases(_read_json(args.cases))
    except BenchmarkError as exc:
        parser.error(str(exc))

    if args.live and args.provider is None:
        parser.error("--provider is required with --live")
    if args.live and args.output is None:
        parser.error("--output is required with --live so results are saved")
    if args.output is not None:
        protected = PROTECTED_OUTPUTS | {args.cases.resolve()}
        if args.output.resolve() in protected:
            parser.error("--output must not overwrite the benchmark fixture or checked-in baseline")

    if not args.live:
        report = dry_run_report(cases, args.provider)
        rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
        if args.output:
            _write_report(args.output, report)
        sys.stdout.write(rendered)
        return 0

    try:
        _load_backend_environment()
    except Exception:
        # A missing dotenv helper does not prevent use of caller-exported environment values.
        pass
    key_available = _provider_key_available(args.provider)
    report = run_live(cases, args.provider, api_key_available=key_available, reasoning_effort=args.reasoning_effort)
    _write_report(args.output, report)
    sys.stdout.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 1 if report["failure_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
