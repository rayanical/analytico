"""Focused coverage for bounded result caches and their safety conditions."""

import json
import sys
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pandas as pd
from fastapi import HTTPException

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from models import AggregateRequest, FilterConfig
from modules import column_interpretation as interpretation
from services import aggregation_service as aggregation
from storage import DatasetInfo
from utils.bounded_cache import BoundedTTLCache


def _luna_response(*, status_code=200, needs_clarification=False):
    decision = {
        "role": "metric",
        "unit": "none",
        "parsing_policy": "parse_decimal",
        "recommended_aggregation": "sum",
        "needs_clarification": needs_clarification,
    }
    return httpx.Response(
        status_code,
        json={
            "status": "completed",
            "model": "gpt-6-luna",
            "usage": {"input_tokens": 12, "output_tokens": 8},
            "output": [{
                "type": "message",
                "content": [{"type": "output_text", "text": json.dumps(decision)}],
            }],
        },
    )


def _interpretation_input(value="10"):
    return {
        "column_name": "measurement",
        "values": [value, "20"],
        "context": {"row_count": 2, "missing_count": 0},
    }


class BoundedCacheTests(unittest.TestCase):
    def test_lru_copying_entry_and_byte_bounds(self):
        cache = BoundedTTLCache[str, dict](max_entries=2, max_bytes=10_000, ttl_seconds=30)
        original = {"items": [1]}
        self.assertTrue(cache.set("a", original))
        original["items"].append(2)
        self.assertEqual(cache.get("a"), {"items": [1]})

        cache.set("b", {"value": 2})
        cache.get("a")  # Promote a, so b becomes the LRU item.
        cache.set("c", {"value": 3})
        self.assertIsNone(cache.get("b"))
        self.assertEqual(cache.get("a"), {"items": [1]})
        self.assertEqual(cache.get("c"), {"value": 3})

        tiny = BoundedTTLCache[str, str](max_entries=2, max_bytes=64, ttl_seconds=30)
        self.assertFalse(tiny.set("large", "x" * 1000))
        self.assertEqual(len(tiny), 0)
        self.assertLessEqual(tiny.size_bytes, tiny.max_bytes)

    def test_ttl_and_concurrent_access_stay_within_limits(self):
        cache = BoundedTTLCache[str, int](max_entries=8, max_bytes=10_000, ttl_seconds=1)
        now = [10.0]
        with patch("utils.bounded_cache.time.monotonic", side_effect=lambda: now[0]):
            cache.set("short-lived", 1)
            now[0] = 12.0
            self.assertIsNone(cache.get("short-lived"))

        def exercise(index):
            for offset in range(20):
                key = str((index + offset) % 16)
                cache.set(key, index + offset)
                cache.get(key)

        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(exercise, range(8)))
        self.assertLessEqual(len(cache), 8)
        self.assertLessEqual(cache.size_bytes, cache.max_bytes)


class InterpretationCacheTests(unittest.TestCase):
    def setUp(self):
        interpretation._INTERPRETATION_CACHE.clear()

    def tearDown(self):
        interpretation._INTERPRETATION_CACHE.clear()

    def test_success_and_clarification_are_cached_with_honest_metadata(self):
        requests = []
        responses = [_luna_response(), _luna_response(needs_clarification=True)]

        class FakeClient:
            def __init__(self, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def post(self, *args, **kwargs):
                requests.append(kwargs)
                return responses.pop(0)

        with patch.dict("os.environ", {
            "COLUMN_INTERPRETER": "luna",
            "COLUMN_INTERPRETER_MODEL": "gpt-6-luna",
            "OPENAI_API_KEY": "secret-key",
        }, clear=True), patch.object(interpretation.httpx, "Client", FakeClient):
            first = interpretation.interpret_column(_interpretation_input(), use_cache=True)
            hit = interpretation.interpret_column(_interpretation_input(), use_cache=True)
            uncertain = interpretation.interpret_column(
                _interpretation_input("unknown"), use_cache=True
            )
            uncertain_hit = interpretation.interpret_column(
                _interpretation_input("unknown"), use_cache=True
            )

        self.assertEqual(len(requests), 2)
        self.assertEqual(first.status, "ok")
        self.assertFalse(first.cache_hit)
        self.assertEqual(hit.status, "ok")
        self.assertTrue(hit.cache_hit)
        self.assertEqual(hit.latency_ms, 0.0)
        self.assertEqual(hit.usage, {})
        self.assertEqual(uncertain.status, "uncertain")
        self.assertEqual(uncertain_hit.status, "uncertain")
        self.assertTrue(uncertain_hit.cache_hit)
        self.assertTrue(all("secret-key" not in key for key in interpretation._INTERPRETATION_CACHE._entries))

    def test_failures_credentials_and_injected_transports_cannot_reuse_success(self):
        calls = []
        responses = [
            _luna_response(status_code=503),
            _luna_response(),
            _luna_response(),
            _luna_response(),
        ]

        class FakeClient:
            def __init__(self, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def post(self, *args, **kwargs):
                calls.append(kwargs)
                return responses.pop(0)

        with patch.dict("os.environ", {
            "COLUMN_INTERPRETER": "luna",
            "COLUMN_INTERPRETER_MODEL": "gpt-6-luna",
            "OPENAI_API_KEY": "first-key",
        }, clear=True), patch.object(interpretation.httpx, "Client", FakeClient):
            failed = interpretation.interpret_column(_interpretation_input(), use_cache=True)
            retried = interpretation.interpret_column(_interpretation_input(), use_cache=True)
            with patch.dict("os.environ", {"OPENAI_API_KEY": "rotated-key"}):
                rotated = interpretation.interpret_column(_interpretation_input(), use_cache=True)
            with patch.dict("os.environ", {"OPENAI_API_KEY": ""}):
                unavailable = interpretation.interpret_column(_interpretation_input(), use_cache=True)
            injected_a = interpretation.interpret_column(
                _interpretation_input(), transport=object(), use_cache=True
            )
            injected_b = interpretation.interpret_column(
                _interpretation_input(), transport=object(), use_cache=True
            )

        self.assertEqual(failed.status, "unavailable")
        self.assertEqual(retried.status, "ok")
        self.assertFalse(retried.cache_hit)
        self.assertFalse(rotated.cache_hit)
        self.assertEqual(unavailable.error_code, "missing_api_key")
        self.assertFalse(injected_a.cache_hit)
        self.assertFalse(injected_b.cache_hit)
        self.assertEqual(len(calls), 5)

    def test_invalid_input_is_rejected_before_any_cache_lookup(self):
        with patch.dict("os.environ", {
            "COLUMN_INTERPRETER": "luna",
            "OPENAI_API_KEY": "key",
        }, clear=True):
            invalid = interpretation.interpret_column(
                {**_interpretation_input(), "extra": "not allowed"}, use_cache=True
            )
        self.assertEqual(invalid.error_code, "invalid_input")


class ChartCacheTests(unittest.TestCase):
    def setUp(self):
        aggregation._CHART_CACHE.clear()

    def tearDown(self):
        aggregation._CHART_CACHE.clear()

    @staticmethod
    def dataset(value=1):
        return SimpleNamespace(
            id="fixture",
            df=pd.DataFrame({"group": ["A", "B"], "value": [value, value + 1]}),
            column_types={"group": "categorical", "value": "metric"},
            column_formats={},
        )

    @staticmethod
    def request(**kwargs):
        fields = {"dataset_id": "fixture", "x_axis_key": "group", "y_axis_keys": ["value"]}
        fields.update(kwargs)
        return AggregateRequest(**fields)

    def test_result_hits_cache_but_dataset_lookup_and_filter_validation_still_run(self):
        ds = self.dataset()
        with patch.object(aggregation, "get_dataset", return_value=ds) as get_dataset, patch.object(
            aggregation, "aggregate_data", wraps=aggregation.aggregate_data
        ) as aggregate_data, patch.object(
            aggregation, "apply_filters", wraps=aggregation.apply_filters
        ) as apply_filters:
            first = aggregation.run_aggregate(self.request())
            second = aggregation.run_aggregate(self.request())
            with self.assertRaises(HTTPException):
                aggregation.run_aggregate(self.request(filters=[FilterConfig(column="missing", values=["x"])]))

        self.assertEqual(first.data, second.data)
        self.assertEqual(aggregate_data.call_count, 1)
        self.assertEqual(get_dataset.call_count, 3)
        self.assertEqual(apply_filters.call_count, 2)

    def test_all_request_fields_and_dataset_replacements_get_distinct_entries(self):
        ds = self.dataset()
        base = self.request()
        requests = [
            self.request(dataset_id="another-fixture"),
            self.request(x_axis_key="value"),
            self.request(y_axis_keys=["group"]),
            self.request(aggregation="mean"),
            self.request(chart_type="line"),
            self.request(filters=[FilterConfig(column="group", values=["A"])]),
            self.request(limit=1),
            self.request(sort_by="label"),
            self.request(group_others=False),
            self.request(include_analysis=True),
            self.request(time_bucket="month"),
        ]
        base_key = aggregation._chart_cache_key(ds, base)
        self.assertEqual(len({base_key, *(aggregation._chart_cache_key(ds, item) for item in requests)}), 12)

        replacement = self.dataset(value=20)
        with patch.object(aggregation, "get_dataset", side_effect=[ds, replacement]), patch.object(
            aggregation, "aggregate_data", wraps=aggregation.aggregate_data
        ) as aggregate_data:
            original = aggregation.run_aggregate(base)
            updated = aggregation.run_aggregate(base)

        self.assertNotEqual(original.data, updated.data)
        self.assertEqual(aggregate_data.call_count, 2)

    def test_expired_or_missing_dataset_is_never_served_from_chart_cache(self):
        ds = self.dataset()
        request = self.request()
        with patch.object(aggregation, "get_dataset", return_value=ds):
            aggregation.run_aggregate(request)
        with patch.object(aggregation, "get_dataset", side_effect=HTTPException(404, "expired")):
            with self.assertRaises(HTTPException):
                aggregation.run_aggregate(request)

    def test_in_place_dataset_edits_require_and_honor_version_invalidation(self):
        ds = DatasetInfo(
            df=pd.DataFrame({"group": ["A", "B"], "value": [1, 2]}),
            filename="fixture.csv",
            cleaning_actions=[],
            missing_counts={},
            column_types={"group": "categorical", "value": "metric"},
            column_formats={},
            profile={},
            default_chart=None,
            suggestions=[],
        )
        request = self.request()
        with patch.object(aggregation, "get_dataset", return_value=ds), patch.object(
            aggregation, "aggregate_data", wraps=aggregation.aggregate_data
        ) as aggregate_data:
            original = aggregation.run_aggregate(request)
            ds.df.loc[:, "value"] = [20, 21]
            ds.invalidate_cached_results()
            updated = aggregation.run_aggregate(request)

        self.assertNotEqual(original.data, updated.data)
        self.assertEqual(aggregate_data.call_count, 2)

    def test_analysis_runs_again_for_a_cached_chart(self):
        ds = self.dataset()
        analysis_message = SimpleNamespace(message=SimpleNamespace(content="Fresh analysis."))
        completions = SimpleNamespace(create=lambda **kwargs: SimpleNamespace(choices=[analysis_message]))
        fake_client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        with patch.dict("os.environ", {"OPENAI_API_KEY": "configured"}), patch.object(
            aggregation, "get_dataset", return_value=ds
        ), patch.object(aggregation, "aggregate_data", wraps=aggregation.aggregate_data) as aggregate_data, patch.object(
            aggregation, "get_openai_client", return_value=fake_client
        ) as get_client:
            first = aggregation.run_aggregate(self.request(include_analysis=True))
            second = aggregation.run_aggregate(self.request(include_analysis=True))

        self.assertEqual(first.analysis, "Fresh analysis.")
        self.assertEqual(second.analysis, "Fresh analysis.")
        self.assertEqual(aggregate_data.call_count, 1)
        self.assertEqual(get_client.call_count, 2)


if __name__ == "__main__":
    unittest.main()
