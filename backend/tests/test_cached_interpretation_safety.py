"""Identical samples never permit an unsafe full column through cached inference."""

import contextlib
import io
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import httpx
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modules import column_interpretation
from modules.data_janitor import _interpretation_input
from services.ingestion_service import ingest_dataframe
from storage import DATASETS


class CachedInterpretationSafetyTests(unittest.TestCase):
    def test_unsampled_contradiction_is_blocked_on_real_cache_hit(self):
        column_interpretation._INTERPRETATION_CACHE.clear()
        self.addCleanup(column_interpretation._INTERPRETATION_CACHE.clear)
        self.addCleanup(DATASETS.clear)
        good = pd.DataFrame({"amount_usd": ["USD 1"] * 24})
        bad = good.copy(deep=True)
        bad.loc[11, "amount_usd"] = "EUR 1"
        self.assertEqual(_interpretation_input("amount_usd", good.amount_usd, list(good.columns)),
                         _interpretation_input("amount_usd", bad.amount_usd, list(bad.columns)))
        decision = {"role": "metric", "unit": "USD", "parsing_policy": "parse_currency_decimal",
                    "recommended_aggregation": "sum", "needs_clarification": False}
        reply = httpx.Response(200, json={"status": "completed", "model": "gpt-6-luna",
            "usage": {"input_tokens": 100, "output_tokens": 20},
            "output": [{"type": "message", "content": [{"type": "output_text", "text": json.dumps(decision)}]}]})
        with patch.dict(os.environ, {"COLUMN_INTERPRETER": "luna", "OPENAI_API_KEY": "synthetic-key",
                                     "COLUMN_INTERPRETER_MODEL": "gpt-6-luna"}), \
             patch.object(column_interpretation.httpx, "Client") as client, \
             patch("services.ingestion_service._generate_business_summary", return_value=None), \
             contextlib.redirect_stdout(io.StringIO()):
            client.return_value.__enter__.return_value.post.return_value = reply
            first = ingest_dataframe(good, "good.csv", "/test")
            second = ingest_dataframe(bad, "bad.csv", "/test")
        self.assertEqual(client.return_value.__enter__.return_value.post.call_count, 1)
        self.assertFalse(first.columns[0].interpretation.cache_hit)
        self.assertEqual(first.columns[0].interpretation.runtime_status, "applied")
        metadata = second.columns[0].interpretation
        self.assertTrue(metadata.cache_hit)
        self.assertEqual(metadata.usage, {})
        self.assertEqual(metadata.runtime_status, "clarification")
        self.assertEqual(second.profile.top_metrics, [])
        self.assertEqual(DATASETS[second.dataset_id].df.amount_usd.tolist(),
                         ["EUR 1" if index == 11 else "USD 1" for index in range(24)])
