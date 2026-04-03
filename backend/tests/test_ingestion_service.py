import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from services.ingestion_service import ingest_dataframe
from storage import DATASETS
from utils.errors import friendly_ingestion_error_message


class IngestionServiceTests(unittest.TestCase):
    def tearDown(self):
        DATASETS.clear()

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
