"""Offline contract checks for the opt-in interpretation runner."""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from evaluate_interpretation import validate_cases
from run_interpretation import dry_run_report, main, run_live


CASES_PATH = Path(__file__).with_name("interpretation_cases.json")


class InterpretationRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = validate_cases(json.loads(CASES_PATH.read_text(encoding="utf-8")))

    def test_default_export_contains_only_routing_id_and_model_input(self):
        report = dry_run_report(self.cases, provider=None)
        self.assertEqual(report["mode"], "dry_run")
        self.assertEqual(len(report["inputs"]), len(self.cases))
        for envelope in report["inputs"]:
            self.assertEqual(set(envelope), {"case_id", "input"})
            self.assertEqual(set(envelope["input"]), {"column_name", "values", "context"})
            self.assertNotIn("category", envelope["input"])
            self.assertNotIn("expected", envelope["input"])

    def test_live_callable_receives_only_projected_input_and_keeps_actual_decision(self):
        case = next(case for case in self.cases if case["id"] == "rate.weighted_with_counts")
        seen = []
        actual_decision = {
            "role": "metric",
            "unit": "ratio",
            "parsing_policy": "parse_percent_to_ratio",
            "recommended_aggregation": "ratio_of_sums",
            "needs_clarification": False,
        }

        def fake_interpret(input_data, *, provider):
            seen.append((input_data, provider))
            return SimpleNamespace(
                decision=actual_decision,
                status="ok",
                model="jev-test-model",
                provider=provider,
                prompt_version="test-v1",
                latency_ms=125.5,
                usage={"input_tokens": 21, "output_tokens": 4},
                confidence=None,
                error_code=None,
            )

        report = run_live([case], "jev", fake_interpret)
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0][1], "jev")
        self.assertEqual(set(seen[0][0]), {"column_name", "values", "context"})
        self.assertEqual(report["results"][0]["decision"], actual_decision)
        self.assertEqual(report["scoring"]["per_field"]["recommended_aggregation"]["correct"], 1)
        self.assertEqual(report["metrics"]["usage"]["tokens"]["total_tokens"]["total"], 25)
        self.assertEqual(report["metrics"]["usage"]["tokens"]["total_tokens"]["derived_cases"], 1)
        self.assertEqual(report["metrics"]["latency_ms"]["samples"], 1)

    def test_missing_key_returns_saved_failure_rows_without_calling_adapter(self):
        cases = self.cases[:2]
        calls = 0

        def should_not_call(*args, **kwargs):
            nonlocal calls
            calls += 1
            raise AssertionError("provider callable must not run without a key")

        report = run_live(cases, "luna", should_not_call, api_key_available=False)
        self.assertEqual(calls, 0)
        self.assertEqual(report["failure_count"], len(cases))
        self.assertTrue(all(row["status"] == "unavailable" for row in report["results"]))
        self.assertTrue(
            all(row["provider_metadata"]["error_code"] == "missing_api_key" for row in report["results"])
        )
        self.assertIsNone(report["metrics"]["usage"]["tokens"]["input_tokens"]["total"])
        self.assertEqual(report["metrics"]["usage"]["cost"]["status"], "unknown")

    def test_call_error_is_sanitized_and_successful_partial_results_are_scored(self):
        calls = 0
        case1, case2 = self.cases[:2]
        partial_decision = {"role": "metric"}

        def fake_interpret(input_data, *, provider):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("provider failed with key sk-secret-value")
            return SimpleNamespace(
                decision=partial_decision,
                status="ok",
                model="luna-test-model",
                provider=provider,
                prompt_version="test-v1",
                latency_ms=300,
                usage={"input_tokens": 8},
                confidence=None,
                error_code=None,
            )

        report = run_live([case1, case2], "luna", fake_interpret)
        self.assertEqual(report["failure_count"], 1)
        self.assertEqual(report["results"][0]["provider_metadata"]["error_code"], "provider_error")
        self.assertNotIn("sk-secret-value", json.dumps(report))
        self.assertIsNone(report["results"][0]["decision"])
        self.assertEqual(report["results"][1]["decision"], partial_decision)
        self.assertEqual(report["scoring"]["predicted_cases"], 1)
        self.assertEqual(report["metrics"]["usage"]["tokens"]["input_tokens"]["total"], 8)

    def test_live_flag_requires_provider_and_output_before_any_environment_or_provider_work(self):
        with patch("run_interpretation._load_backend_environment") as load_env:
            with self.assertRaises(SystemExit):
                main(["--live"])
        load_env.assert_not_called()

    def test_live_cli_saves_missing_key_report_without_loading_adapter(self):
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "missing-key-report.json"
            stdout = io.StringIO()
            with (
                patch.dict("os.environ", {}, clear=True),
                patch("run_interpretation._load_backend_environment"),
                patch(
                    "run_interpretation._load_interpret_column",
                    side_effect=AssertionError("adapter must not load without a key"),
                ),
                contextlib.redirect_stdout(stdout),
            ):
                exit_code = main(
                    [
                        "--live",
                        "--provider",
                        "jev",
                        "--cases",
                        str(CASES_PATH),
                        "--output",
                        str(output_path),
                    ]
                )
            report = json.loads(output_path.read_text(encoding="utf-8"))
        self.assertEqual(exit_code, 1)
        self.assertEqual(report["failure_count"], len(self.cases))
        self.assertTrue(all(row["decision"] is None for row in report["results"]))
        self.assertIn('"missing_api_key"', stdout.getvalue())

    def test_default_cli_path_is_dry_run_and_never_imports_adapter(self):
        stdout = io.StringIO()
        with patch("run_interpretation._load_interpret_column", side_effect=AssertionError("must stay offline")):
            with contextlib.redirect_stdout(stdout):
                exit_code = main(["--cases", str(CASES_PATH)])
        report = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(report["mode"], "dry_run")


if __name__ == "__main__":
    unittest.main()
