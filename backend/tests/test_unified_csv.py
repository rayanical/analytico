"""Full ingestion contracts that must hold when small CSVs use DuckDB."""
import io
import os
import unittest
from unittest.mock import patch

from fastapi import HTTPException
import pandas as pd

from modules.import_policy import ImportSettings, CSVStructureError
from modules.data_janitor import _unique_normalized_headers
from services.csv_ingestion import ingest_csv
from storage import DATASETS


class UnifiedCSVTests(unittest.TestCase):
    def setUp(self):
        environment = patch.dict(os.environ, {'OPENAI_API_KEY':'', 'AI_GATEWAY_API_KEY':'',
                                             'COLUMN_INTERPRETER':'off', 'ANALYTICO_NATIVE_VALIDATION':'0'})
        environment.start()
        self.addCleanup(environment.stop)

    def outcome(self, source, engine, settings=None, overrides=None):
        ds = None
        try:
            response = ingest_csv(io.BytesIO(source), 'contract.csv', '/test', engine=engine,
                                  enqueue_enrichment=False, import_settings=settings,
                                  column_overrides=overrides)
            ds = DATASETS[response.dataset_id]
            frame = ds.sample_frame(1000)
            values = {c:[None if pd.isna(v) else v for v in frame[c]] for c in frame.columns}
            path = ds.disk.source_path if hasattr(ds, 'disk') else ds.source_path
            self.assertEqual(path.read_bytes(), source)
            missing = dict(zip(_unique_normalized_headers(list(ds.raw_missing_counts)),
                               ds.raw_missing_counts.values()))
            return {'rows':ds.row_count, 'values':values, 'roles':ds.column_types,
                    'formats':ds.column_formats, 'missing':missing}
        except HTTPException as error:
            return {'error_status':error.status_code}
        except CSVStructureError as error:
            return {'error_status':413 if error.limit_exceeded else 400}
        finally:
            if ds is not None:
                DATASETS.pop(ds.id, None)
                ds.close()

    def test_supported_dialects_encodings_nulls_and_numeric_values_match(self):
        cases = [
            (b'Code,Code,Label\nNA,001,NA\nNULL,002,NULL\nN/A,003,N/A\n,004,\n', None),
            ('code,label\n001,"caf\u00e9, \"\"hello\"\""\n002,"line one\nline two"\n'.encode(), None),
            (b'\ncode,label\n001,A\n002,B\n', None),
            (b'label\nA\n\n   \n"   "\n""\nB\n', None),
            ('code,label\n001,caf\u00e9\n'.encode('utf-16'), ImportSettings(encoding='utf-16')),
            ('code,label\n001,caf\u00e9\n'.encode('cp1252'), ImportSettings(encoding='cp1252')),
            ('code,label\n001,caf\u00e9\n'.encode('utf-8-sig'), None),
            (b'code|amount\n001|12\n002|34\n', ImportSettings(delimiter='|')),
            (b'label\tamount\nA\t12\nB\t34\n', ImportSettings(delimiter='\t')),
            (b'code,label\nNA,NULL\nN/A,\n001,NA\n', ImportSettings(null_values=['', 'NA', 'NULL'])),
            (b'code,label\n001,\n002,NA\n', ImportSettings(null_values=[])),
            (b'group,amount\nA,$9007199254740992\nA,$9007199254740993\n', None),
            (b'group,amount\nA,18446744073709551614\nB,18446744073709551615\n', None),
            (b'group,amount\nA,9007199254740993.25\nB,1e309\n', None),
            (b'code,amount\n001,0012\n002,0013\n', None),
            ('group,amount\nA,\u20ac12.50\nB,$15.75\n'.encode(), None),
            (b'group;amount\nA;1.234,56\nB;2.345,67\n',
             ImportSettings(delimiter=';', decimal_separator=',', grouping_separator='.')),
        ]
        for source, settings in cases:
            with self.subTest(source=source[:45], settings=settings):
                self.assertEqual(self.outcome(source, 'disk', settings),
                                 self.outcome(source, 'pandas', settings))

    def test_malformed_sources_reject_consistently(self):
        for source in [b'', b'code,label\n', b'code,label\n001,A,extra\n',
                       b'code,label\n001\n', b'code,label\n001,"unclosed\n',
                       b'code,label\n001,"A" \n', b'code,label\n001,A\x00B\n']:
            with self.subTest(source=source):
                self.assertEqual(self.outcome(source, 'disk'), self.outcome(source, 'pandas'))

    def test_explicit_text_and_identifier_policies_preserve_lexemes(self):
        source = b'code,value\n001,12.50\n002,31.75\n'
        overrides = [{'column':'value','parse_as':'text'}, {'column':'code','role':'identifier'}]
        self.assertEqual(self.outcome(source, 'disk', overrides=overrides),
                         self.outcome(source, 'pandas', overrides=overrides))

    def test_auto_uses_disk_even_for_tiny_files(self):
        with patch.dict(os.environ, {'ANALYTICO_INGESTION_ENGINE':'auto',
                                   'ANALYTICO_DISK_THRESHOLD_BYTES':'9999999999'}):
            response = ingest_csv(io.BytesIO(b'group,amount\nA,12\n'), 'tiny.csv', '/test',
                                  enqueue_enrichment=False)
        ds = DATASETS.pop(response.dataset_id)
        try:
            self.assertTrue(hasattr(ds, 'disk'))
            self.assertEqual(ds.row_count, 1)
        finally:
            ds.close()

    def test_application_preloads_engine_before_serving(self):
        from fastapi.testclient import TestClient
        from main import app
        with patch('main.preload_ingestion_engine') as preload:
            with TestClient(app) as client:
                preload.assert_called_once()
                self.assertEqual(client.get('/').status_code, 200)


if __name__ == '__main__':
    unittest.main()
