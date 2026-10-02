"""Shared column statistics preserve ingestion results while avoiding rescans."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from modules.column_statistics import compute_column_statistics
from modules.intelligence import (
    SemanticType,
    auto_profile,
    detect_semantic_type,
    generate_default_chart,
)
from services.response_builders import build_upload_response, get_column_summary
from storage import DatasetInfo


def _frame():
    return pd.DataFrame(
        {
            "period_year": [2020, 2021] * 26,
            "revenue": [float(index) for index in range(52)],
            "region": (["Zulu", "alpha", "Zulu", "beta"] * 13),
            "reference": [f"ref-{index}" for index in range(51)] + ["ref-0"],
        }
    )


class ColumnStatisticsTests(unittest.TestCase):
    def test_cached_and_uncached_ingestion_results_match(self):
        frame = _frame()
        column_stats = compute_column_statistics(frame)

        uncached_types = {
            column: detect_semantic_type(frame, column)
            for column in frame.columns
        }
        cached_types = {
            column: detect_semantic_type(frame, column, column_stats)
            for column in frame.columns
        }
        self.assertEqual(cached_types, uncached_types)
        self.assertEqual(cached_types["period_year"], SemanticType.TEMPORAL)

        uncached_chart = generate_default_chart(frame, uncached_types)
        cached_chart = generate_default_chart(frame, cached_types, column_stats)
        self.assertEqual(cached_chart, uncached_chart)

        uncached_profile = auto_profile(frame, cached_types)
        profile = auto_profile(frame, cached_types, column_stats=column_stats)
        self.assertEqual(profile, uncached_profile)
        ds_info = DatasetInfo(
            df=frame,
            filename="sample.csv",
            cleaning_actions=[],
            missing_counts={},
            column_types=cached_types,
            column_formats={column: "general" for column in frame.columns},
            profile=profile,
            default_chart=cached_chart,
            suggestions=[],
        )
        common = dict(
            ds_info=ds_info,
            df=frame,
            col_types=cached_types,
            col_formats={column: "general" for column in frame.columns},
            missing_counts={},
            cleaning_actions=[],
            quality=100.0,
            profile=profile,
            default_chart=cached_chart,
            suggestions=[],
            summary=None,
        )
        uncached_response = build_upload_response(**common)
        cached_response = build_upload_response(**common, column_stats=column_stats)
        self.assertEqual(cached_response.model_dump(), uncached_response.model_dump())

        reference = column_stats["reference"]
        self.assertEqual(reference.unique_count, 51)
        self.assertEqual(reference.sample_values, tuple(f"ref-{index}" for index in range(20)))
        self.assertEqual(len(reference.sample_values), 20)

    def test_empty_and_extension_dtype_columns_match_uncached_results(self):
        empty = pd.DataFrame(
            {
                "year": pd.Series([], dtype="int64"),
                "label": pd.Series([], dtype="object"),
                "category": pd.Series(pd.Categorical([], categories=["unused"])),
            }
        )
        nulls_and_extensions = pd.DataFrame(
            {
                "nullable_count": pd.Series([1, pd.NA, 2, 2], dtype="Int64"),
                "category": pd.Series(
                    pd.Categorical(
                        ["b", None, "a", "b"], categories=["unused", "b", "a"]
                    )
                ),
                "when": pd.to_datetime(
                    ["2025-01-01T00:00:00Z", None, "2025-01-02T00:00:00Z", "2025-01-01T00:00:00Z"],
                    utc=True,
                ),
                "all_null": pd.Series([pd.NA] * 4, dtype="string"),
            }
        )

        for frame in (empty, nulls_and_extensions):
            with self.subTest(columns=list(frame.columns)):
                column_stats = compute_column_statistics(frame)
                cached_types = {
                    column: detect_semantic_type(frame, column, column_stats)
                    for column in frame.columns
                }
                uncached_types = {
                    column: detect_semantic_type(frame, column)
                    for column in frame.columns
                }
                self.assertEqual(cached_types, uncached_types)
                self.assertEqual(
                    generate_default_chart(frame, cached_types, column_stats),
                    generate_default_chart(frame, uncached_types),
                )
                self.assertEqual(
                    auto_profile(frame, cached_types, column_stats=column_stats),
                    auto_profile(frame, uncached_types),
                )

                for column in frame.columns:
                    cached_summary = get_column_summary(
                        frame, column, cached_types[column], "general", column_stats
                    )
                    uncached_summary = get_column_summary(
                        frame, column, uncached_types[column], "general"
                    )
                    self.assertEqual(cached_summary.model_dump(), uncached_summary.model_dump())
                    self.assertEqual(column_stats[column].unique_count, int(frame[column].nunique()))

        self.assertEqual(compute_column_statistics(empty)["label"].unique_count, 0)
        self.assertEqual(nulls_and_extensions["category"].nunique(), 2)
        self.assertEqual(compute_column_statistics(nulls_and_extensions)["category"].unique_count, 2)
        self.assertEqual(compute_column_statistics(nulls_and_extensions)["all_null"].sample_values, ())

    def test_cached_detection_chart_and_summary_do_not_rescan_distinct_values(self):
        frame = _frame()
        original_unique = pd.Series.unique
        original_map = pd.Series.map
        calls = {}
        map_calls = {}

        def count_unique(series, *args, **kwargs):
            calls[series.name] = calls.get(series.name, 0) + 1
            return original_unique(series, *args, **kwargs)

        def count_map(series, *args, **kwargs):
            map_calls[series.name] = map_calls.get(series.name, 0) + 1
            return original_map(series, *args, **kwargs)

        with patch.object(pd.Series, "unique", count_unique), patch.object(
            pd.Series, "map", count_map
        ):
            column_stats = compute_column_statistics(frame)
            column_types = {
                column: detect_semantic_type(frame, column, column_stats)
                for column in frame.columns
            }
            chart = generate_default_chart(frame, column_types, column_stats)
            auto_profile(frame, column_types, column_stats=column_stats)
            summary = get_column_summary(
                frame, "reference", column_types["reference"], "general", column_stats
            )

        self.assertIsNotNone(chart)
        self.assertEqual(summary.unique_count, 51)
        self.assertEqual(len(summary.sample_values), 20)
        self.assertEqual(calls, {column: 1 for column in frame.columns})
        self.assertEqual(map_calls, {"period_year": 1})

    def test_statistics_mapping_and_records_are_immutable(self):
        column_stats = compute_column_statistics(_frame())

        with self.assertRaises(TypeError):
            column_stats["new"] = column_stats["region"]
        with self.assertRaises(AttributeError):
            column_stats["region"].unique_count = 1


if __name__ == "__main__":
    unittest.main()
