import sys
import io
import unittest
import os
from pathlib import Path
from unittest.mock import patch

import pandas as pd

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from services.ingestion_service import ingest_dataframe
from storage import DATASETS
from utils.errors import friendly_ingestion_error_message
from utils.dataframe_utils import read_csv_fast


class IngestionServiceTests(unittest.TestCase):
    def setUp(self):
        self.no_ai = patch.dict(os.environ, {"OPENAI_API_KEY": ""})
        self.no_ai.start()

    def tearDown(self):
        DATASETS.clear()
        self.no_ai.stop()

    def test_csv_retains_literal_na_tokens_and_blank_missing_values(self):
        frame = read_csv_fast(io.StringIO("code,label\nNA,NA\nNULL,NULL\nN/A,N/A\n001,\n"))
        self.assertEqual(frame["code"].tolist(), ["NA", "NULL", "N/A", "001"])
        self.assertEqual(frame["label"].iloc[:3].tolist(), ["NA", "NULL", "N/A"])
        self.assertTrue(pd.isna(frame["label"].iloc[3]))
        response = ingest_dataframe(frame, "tokens.csv", endpoint_name="/upload")
        self.assertEqual(DATASETS[response.dataset_id].raw_df["code"].tolist(), ["NA", "NULL", "N/A", "001"])

    def test_ingest_dataframe_returns_upload_response_shape(self):
        df = pd.DataFrame({"revenue": [10, 20, 30], "category": ["A", "B", "A"]})

        with patch("services.ingestion_service._generate_business_summary", return_value="Test summary"):
            response = ingest_dataframe(df, "sample.csv", endpoint_name="/upload")

        payload = response.model_dump()
        self.assertIn("dataset_id", payload)
        self.assertEqual(payload["filename"], "sample.csv")
        self.assertEqual(payload["row_count"], 3)
        self.assertIn("columns", payload)
        self.assertIn("data_health", payload)
        self.assertIn("profile", payload)
        self.assertIn("suggestions", payload)

    def test_ingestion_retains_an_immutable_raw_dataframe(self):
        source = pd.DataFrame({"Revenue?": ["1.25", "bad", None]})
        expected_raw = source.copy(deep=True)
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}):
            response = ingest_dataframe(source, "raw.csv", endpoint_name="/upload")
        dataset = DATASETS[response.dataset_id]
        pd.testing.assert_frame_equal(dataset.raw_df, expected_raw)
        self.assertIsNot(dataset.raw_df, source)
        self.assertEqual(dataset.raw_missing_counts, {"Revenue?": 1})
        self.assertEqual(response.data_health.missing_values, {"revenue": 1})
        self.assertEqual(dataset.df["revenue"].tolist()[:2], ["1.25", "bad"])

    def test_missing_ai_key_skips_summary_and_keeps_manual_ingestion_available(self):
        df = pd.DataFrame({"amount": [10, 20, 30]})
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}), patch(
            "core.config.get_openai_client", side_effect=AssertionError("AI client should not be initialized")
        ):
            response = ingest_dataframe(df, "manual.csv", endpoint_name="/upload")
        self.assertIsNone(response.summary)
        self.assertEqual(response.profile.row_count, 3)

    def test_failed_response_construction_does_not_store_dataset(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}), patch(
            "services.ingestion_service.build_upload_response", side_effect=RuntimeError("response failed")
        ):
            with self.assertRaisesRegex(RuntimeError, "response failed"):
                ingest_dataframe(pd.DataFrame({"value": [1, 2]}), "failed.csv", endpoint_name="/upload")
        self.assertEqual(DATASETS, {})

    def test_friendly_ingestion_error_message_mapping(self):
        status, detail = friendly_ingestion_error_message(
            ValueError("could not convert string to float: 'bad-value'")
        )
        self.assertEqual(status, 400)
        self.assertIn("auto-parse", detail)

        generic_status, generic_detail = friendly_ingestion_error_message(RuntimeError("boom"))
        self.assertEqual(generic_status, 500)
        self.assertIn("Dataset processing failed", generic_detail)


if __name__ == "__main__":
    unittest.main()
