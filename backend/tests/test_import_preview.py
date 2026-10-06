"""The user reviews a bounded sample; confirmation validates the full source."""

import io
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastapi.testclient import TestClient
from main import app
from services import import_preview
from storage import DATASETS, get_dataset


class ImportPreviewTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {"OPENAI_API_KEY": "", "COLUMN_INTERPRETER": "off",
                                                   "ANALYTICO_INGESTION_ENGINE": "pandas"})
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.client = TestClient(app)
        self.addCleanup(self.cleanup)
        self.cleanup()

    def cleanup(self):
        for item in list(import_preview._imports.values()):
            import_preview.cancel_import(item.id)
        for dataset in list(DATASETS.values()):
            dataset.close()
        DATASETS.clear()

    def stage(self, text, **kwargs):
        response = self.client.post("/imports/preview", files={"file": ("sample.csv", text.encode(), "text/csv")}, **kwargs)
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def confirm(self, preview, **kwargs):
        return self.client.post(f"/imports/{preview['import_id']}/confirm",
                                json={"settings": preview["settings"], "column_overrides": [], **kwargs})

    def test_preview_is_local_and_not_a_registered_dataset_then_confirm_retains_source(self):
        text = "Code,Amount\n001,12\n002,34\n"
        with patch("services.ingestion_service._generate_business_summary", side_effect=AssertionError("preview must not use AI")):
            preview = self.stage(text)
        self.assertTrue(preview["can_confirm"])
        self.assertEqual(preview["raw_rows"][0][0], "001")
        self.assertEqual(len(DATASETS), 0)
        response = self.confirm(preview)
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertTrue(result["version"])
        self.assertTrue(result["column_schema"])
        dataset = get_dataset(result["dataset_id"])
        self.assertEqual(dataset.source_path.read_text(), text)
        self.assertEqual(dataset.raw_df.iloc[0, 0], "001")
        self.assertEqual(self.confirm(preview).status_code, 404)

    def test_ai_columns_are_opt_in_for_both_ingestion_engines(self):
        from modules.column_interpretation import InterpretationResult
        results = []

        def run_job(dataset_id, version, work):
            results.append(work())
            return {"status": "done"}

        for engine in ("pandas", "disk"):
            for enabled in (False, True):
                with self.subTest(engine=engine, enabled=enabled), \
                     patch.dict(os.environ, {"OPENAI_API_KEY": "offline-test", "COLUMN_INTERPRETER": "off",
                                             "ANALYTICO_INGESTION_ENGINE": engine}), \
                     patch("services.enrichment_service.enqueue_enrichment", side_effect=run_job), \
                     patch("services.ingestion_service._generate_business_summary", return_value="Summary"), \
                     patch("modules.schema_interpretation.analyze_schema", return_value={
                         "interpretation_proposals": {c: {"status": "uncertain", "decision": None} for c in ("group", "amount")},
                         "coverage": {"complete": True}}) as interpret:
                    preview = self.stage("group,amount\nA,12\nB,34\n")
                    # Omitted flag must retain local column detection.
                    response = self.confirm(preview, **({"ai_column_analysis": True} if enabled else {}))
                    self.assertEqual(response.status_code, 200, response.text)
                    self.assertEqual(interpret.call_count, 1 if enabled else 0)
                    self.assertEqual(len(results[-1]["interpretation_proposals"]), 2 if enabled else 0)
                    self.assertEqual(results[-1]["summary"], "Summary")
                    if not enabled:
                        self.assertIsNone(results[-1]["coverage"])
                    dataset = get_dataset(response.json()["dataset_id"])
                    self.assertEqual(dataset.column_types["amount"], "metric")

    def test_semicolon_decimal_comma_preview_and_confirmation(self):
        preview = self.stage("group;Amount\nA;1.234,56\nB;2.000,50\n")
        self.assertEqual(preview["settings"]["delimiter"], ";")
        self.assertEqual([column["name"] for column in preview["columns"]], ["group", "amount"])
        settings = {**preview["settings"], "decimal_separator": ",", "grouping_separator": "."}
        self.assertEqual(self.confirm(preview, settings=settings).status_code, 409)
        revised = self.client.post(f"/imports/{preview['import_id']}/preview", json={"settings": settings}).json()
        self.assertTrue(revised["can_confirm"], revised)
        self.assertEqual(float(revised["parsed_rows"][0]["amount"]), 1234.56)
        response = self.confirm(revised)
        self.assertEqual(response.status_code, 200, response.text)
        records = self.client.post("/drilldown", json={"dataset_id": response.json()["dataset_id"], "limit": 10}).json()["data"]
        self.assertEqual(records[0]["amount"], 1234.56)
        self.assertEqual(records[1]["amount"], 2000.5)

    def test_late_bad_record_is_not_hidden_by_good_preview(self):
        text = "group,amount\n" + "A,1\n" * 40 + "B,2,unexpected\n"
        preview = self.stage(text)
        self.assertTrue(preview["can_confirm"])
        self.assertFalse(preview["sample_complete"])
        response = self.confirm(preview)
        self.assertEqual(response.status_code, 400, response.text)
        self.assertIn("row", response.json()["detail"].lower())
        self.assertEqual(len(DATASETS), 0)
        self.assertIn(preview["import_id"], import_preview._imports)

    def test_wrong_delimiter_is_correctable_without_reupload(self):
        preview = self.stage("group;amount\nA;1234,56\nB;5678,90\n")
        settings = {**preview["settings"], "delimiter": ","}
        bad = self.client.post(f"/imports/{preview['import_id']}/preview", json={"settings": settings})
        self.assertEqual(bad.status_code, 200)
        self.assertFalse(bad.json()["can_confirm"])
        self.assertEqual(self.confirm(bad.json()).status_code, 409)
        settings["delimiter"] = ";"
        fixed = self.client.post(f"/imports/{preview['import_id']}/preview", json={"settings": settings}).json()
        self.assertTrue(fixed["can_confirm"])

    def test_one_column_quoted_newlines_duplicates_and_literal_nan_are_preserved(self):
        for text in ('note\n"line one\nline two"\nnan\n', 'Value,Value\n1,2\n'):
            preview = self.stage(text)
            self.assertTrue(preview["can_confirm"], preview)
            self.assertEqual(self.confirm(preview).status_code, 200)
        self.assertEqual(preview["columns"][0]["original_name"], "Value")
        self.assertEqual(preview["columns"][1]["original_name"], "Value")

    def test_cancel_ttl_and_capacity_release_owned_files(self):
        preview = self.stage("value\nA\n")
        item = import_preview._imports[preview["import_id"]]
        self.client.delete(f"/imports/{item.id}")
        self.assertFalse(item.path.exists())
        self.assertEqual(self.confirm(preview).status_code, 404)
        preview = self.stage("value\nB\n")
        item = import_preview._imports[preview["import_id"]]
        item.touched -= import_preview.IMPORT_TTL_SECONDS + 1
        response = self.client.post(f"/imports/{item.id}/preview", json={"settings": preview["settings"]})
        self.assertEqual(response.status_code, 404)
        self.assertFalse(item.path.exists())
        with patch.object(import_preview, "MAX_STAGED_IMPORTS", 1):
            preview = self.stage("value\nC\n")
            response = self.client.post("/imports/preview", files={"file": ("second.csv", b"value\nD\n", "text/csv")})
            self.assertEqual(response.status_code, 429)
            self.client.delete(f"/imports/{preview['import_id']}")
            self.stage("value\nD\n")

    def test_direct_upload_rejects_implicit_index_field_loss(self):
        response = self.client.post("/upload", files={"file": ("bad.csv", b"name;amount\nAlice;1234,56\n", "text/csv")})
        self.assertEqual(response.status_code, 400, response.text)
        self.assertEqual(len(DATASETS), 0)


if __name__ == "__main__":
    unittest.main()
