"""Reviewed policies must agree across engines and preserve failed versions."""
import io
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastapi import HTTPException
from modules.import_policy import ColumnOverride, ImportSettings
from services.csv_ingestion import ingest_csv
from services.schema_review import apply_schema, get_schema
from storage import DATASETS, get_dataset, lease_dataset


class SchemaReviewTests(unittest.TestCase):
    def test_saved_file_settings_can_be_edited_and_failed_rebuild_keeps_version(self):
        for engine in ['pandas', 'disk']:
            with self.subTest(engine=engine), patch.dict(os.environ, {'ANALYTICO_INGESTION_ENGINE': engine}):
                old, _ = self.ingest('group;amount;label\nA;1.234,56;NULL\nB;2.000,50;NA\n', engine,
                    ImportSettings(delimiter=';'))
                changed = apply_schema(old.dataset_id, old.version, [],
                    ImportSettings(delimiter=';', decimal_separator=',', grouping_separator='.', null_values=['', 'NULL']))
                frame = get_dataset(changed.dataset_id).sample_frame(10)
                self.assertEqual(frame['amount'].tolist(), [1234.56, 2000.5])
                self.assertEqual(frame['label'].iloc[1], 'NA')
                self.assertEqual(get_schema(changed.dataset_id)['settings']['decimal_separator'], ',')
                with self.assertRaises(HTTPException):
                    apply_schema(changed.dataset_id, changed.version, [ColumnOverride(column='amount', parse_as='number')],
                        ImportSettings(delimiter=';', decimal_separator='.'))
                self.assertEqual(get_schema(changed.dataset_id)['version'], changed.version)

    def test_separator_change_never_reuses_column_overrides_on_new_fields(self):
        for engine in ['pandas', 'disk']:
            with self.subTest(engine=engine), patch.dict(os.environ, {'ANALYTICO_INGESTION_ENGINE': engine}):
                old, _ = self.ingest('code;amount\n001;12\n002;34\n', engine,
                    overrides=[ColumnOverride(column='code_amount', parse_as='text')])
                with self.assertRaises(HTTPException) as conflicting:
                    apply_schema(old.dataset_id, old.version, [ColumnOverride(column='code_amount', role='identifier')],
                        ImportSettings(delimiter=';'))
                self.assertEqual(conflicting.exception.status_code, 422)
                changed = apply_schema(old.dataset_id, old.version, [], ImportSettings(delimiter=';'))
                frame = get_dataset(changed.dataset_id).sample_frame(10)
                self.assertEqual(frame.columns.tolist(), ['code', 'amount'])
                self.assertEqual(frame['code'].tolist(), ['001', '002'])
                self.assertEqual(get_dataset(changed.dataset_id).column_overrides, [])

    def setUp(self):
        env = patch.dict(os.environ, {'OPENAI_API_KEY': '', 'COLUMN_INTERPRETER': 'off'})
        env.start()
        self.addCleanup(env.stop)
        self.addCleanup(self.cleanup)

    def cleanup(self):
        for dataset in list(DATASETS.values()):
            dataset.close()
        DATASETS.clear()

    def ingest(self, text, engine, settings=None, overrides=None):
        result = ingest_csv(io.BytesIO(text.encode()), 'review.csv', '/test', engine=engine,
                            import_settings=settings, column_overrides=overrides,
                            defer_enrichment=True, enqueue_enrichment=False)
        return result, get_dataset(result.dataset_id).sample_frame(200)

    def test_locale_identifiers_dates_and_nulls_match_both_engines(self):
        source = 'Code;amount;when;label\n001;1.234,56;02/03/2025;NA\n002;2.000,50;04/05/2025;NULL\n'
        settings = ImportSettings(delimiter=';', decimal_separator=',', grouping_separator='.', date_order='dmy', null_values=['', 'NULL'])
        frames = []
        for engine in ['pandas', 'disk']:
            with self.subTest(engine=engine):
                _, frame = self.ingest(source, engine, settings)
                self.assertEqual(frame['code'].tolist(), ['001', '002'])
                self.assertEqual(frame['amount'].tolist(), [1234.56, 2000.5])
                self.assertEqual(str(frame['when'].iloc[0])[:10], '2025-03-02')
                self.assertEqual(frame['label'].iloc[0], 'NA')
                frames.append(frame)
        self.assertEqual(frames[0]['when'].tolist(), frames[1]['when'].tolist())

    def test_confirmed_currency_and_percent_parse_original_lexemes(self):
        for engine in ['pandas', 'disk']:
            with self.subTest(engine=engine):
                _, frame = self.ingest('group,amount,rate\nA,$12.50,12%\nB,$2.50,25%\n', engine,
                    overrides=[ColumnOverride(column='amount', format='currency', unit='USD'),
                               ColumnOverride(column='rate', format='percentage', aggregation='mean')])
                self.assertEqual(frame['amount'].tolist(), [12.5, 2.5])
                self.assertEqual(frame['rate'].tolist(), [0.12, 0.25])

    def test_timezone_and_high_precision_dates_preserve_source(self):
        for value in ['2025-01-02T03:04:05Z', '2025-01-02T03:04:05.123456789']:
            for engine in ['pandas', 'disk']:
                with self.subTest(value=value, engine=engine):
                    _, frame = self.ingest(f'group,when\nA,{value}\nB,{value}\n', engine)
                    self.assertEqual(frame['when'].tolist(), [value, value])

    def test_date_values_after_initial_missing_rows_are_not_hidden(self):
        source = 'group,when\n' + 'A,\n' * 120 + 'B,2025-01-02\n'
        for engine in ['pandas', 'disk']:
            with self.subTest(engine=engine):
                _, frame = self.ingest(source, engine, overrides=[ColumnOverride(column='when', parse_as='date')])
                self.assertEqual(str(frame['when'].iloc[-1])[:10], '2025-01-02')

    def test_schema_apply_patch_reset_and_stale_version_are_atomic(self):
        for engine in ['pandas', 'disk']:
            with self.subTest(engine=engine), patch.dict(os.environ, {'ANALYTICO_INGESTION_ENGINE': engine}):
                old, _ = self.ingest('group,amount,code\nA,12,001\nB,34,002\n', engine)
                with lease_dataset(old.dataset_id) as leased:
                    source = getattr(getattr(leased, 'disk', None), 'source_path', None) or leased.source_path
                    source_bytes = source.read_bytes()
                    changed = apply_schema(old.dataset_id, old.version, [ColumnOverride(column='amount', parse_as='text')])
                    self.assertEqual(changed.dataset_id, old.dataset_id)
                    self.assertNotEqual(changed.version, old.version)
                    self.assertEqual(source.read_bytes(), source_bytes)
                current = get_schema(old.dataset_id)
                confirmed = [item['column'] for item in current['columns'] if item['status'] == 'confirmed']
                self.assertEqual(confirmed, ['amount'])
                with self.assertRaises(HTTPException) as stale:
                    apply_schema(old.dataset_id, old.version, [])
                self.assertEqual(stale.exception.status_code, 409)
                with self.assertRaises(HTTPException) as invalid:
                    apply_schema(old.dataset_id, changed.version, [ColumnOverride(column='code', parse_as='date')])
                self.assertEqual(invalid.exception.status_code, 400)
                self.assertEqual(get_schema(old.dataset_id)['version'], changed.version)
                reset = apply_schema(old.dataset_id, changed.version, [ColumnOverride(column='amount', parse_as='auto', role=None, format=None, unit=None, aggregation=None)])
                self.assertEqual(get_dataset(reset.dataset_id).sample_frame(10)['amount'].tolist(), [12, 34])
                self.assertEqual(get_dataset(reset.dataset_id).column_overrides, [])

    def test_invalid_mixed_units_do_not_replace_active_dataset(self):
        for source, column_format in [('group,amount\nA,$12\nB,€14\n', 'currency'), ('group,amount\nA,12%\nB,14\n', 'percentage')]:
            for engine in ['pandas', 'disk']:
                with self.subTest(engine=engine, format=column_format), patch.dict(os.environ, {'ANALYTICO_INGESTION_ENGINE': engine}):
                    old, _ = self.ingest(source, engine)
                    with self.assertRaises(HTTPException):
                        apply_schema(old.dataset_id, old.version, [ColumnOverride(column='amount', format=column_format)])
                    self.assertEqual(get_schema(old.dataset_id)['version'], old.version)

    def test_unknown_duplicate_columns_and_conflicting_roles_are_rejected(self):
        for engine in ['pandas', 'disk']:
            with self.subTest(engine=engine), patch.dict(os.environ, {'ANALYTICO_INGESTION_ENGINE': engine}):
                old, _ = self.ingest('group,amount\nA,12\nB,34\n', engine)
                for overrides, status in [([ColumnOverride(column='missing')], 422),
                       ([ColumnOverride(column='amount'), ColumnOverride(column='amount')], 422),
                       ([ColumnOverride(column='amount', parse_as='number', role='identifier')], 400)]:
                    with self.assertRaises(HTTPException) as error:
                        apply_schema(old.dataset_id, old.version, overrides)
                    self.assertEqual(error.exception.status_code, status)
                self.assertEqual(get_schema(old.dataset_id)['version'], old.version)

    def test_encoding_duplicate_headers_and_quoted_newlines_match_engines(self):
        source = 'Code;Code;Label\n001;002;"café\nsecond line"\n003;004;NULL\n'
        for encoding in ['cp1252', 'utf-16']:
            for engine in ['pandas', 'disk']:
                with self.subTest(encoding=encoding, engine=engine):
                    result = ingest_csv(io.BytesIO(source.encode(encoding)), 'encoded.csv', '/test', engine=engine,
                        import_settings=ImportSettings(delimiter=';', encoding=encoding),
                        defer_enrichment=True, enqueue_enrichment=False)
                    frame = get_dataset(result.dataset_id).sample_frame(10)
                    self.assertEqual(frame.columns.tolist(), ['code', 'code_1', 'label'])
                    self.assertEqual(frame['code_1'].tolist(), ['002', '004'])
                    self.assertEqual(frame['label'].tolist(), ['café\nsecond line', 'NULL'])
                    self.assertEqual([item['original_name'] for item in result.column_schema], ['Code', 'Code', 'Label'])

    def test_ambiguous_dates_require_a_selected_order_for_forced_parsing(self):
        for engine in ['pandas', 'disk']:
            with self.subTest(engine=engine):
                with self.assertRaises((ValueError, HTTPException)):
                    self.ingest('group,when\nA,02/03/2025\nB,04/05/2025\n', engine,
                        overrides=[ColumnOverride(column='when', parse_as='date')])
                _, frame = self.ingest('group,when\nA,02/03/2025\nB,04/05/2025\n', engine,
                    settings=ImportSettings(date_order='mdy'),
                    overrides=[ColumnOverride(column='when', parse_as='date')])
                self.assertEqual(str(frame['when'].iloc[0])[:10], '2025-02-03')

    def test_unit_only_edits_do_not_strip_locale_identifier_zeros(self):
        for engine in ['pandas', 'disk']:
            with self.subTest(engine=engine):
                _, frame = self.ingest('code;amount\n001;1,25\n002;2,50\n', engine,
                    settings=ImportSettings(delimiter=';', decimal_separator=','),
                    overrides=[ColumnOverride(column='code', unit='reference')])
                self.assertEqual(frame['code'].tolist(), ['001', '002'])

    def test_currency_codes_cannot_conflict_with_symbols(self):
        for engine in ['pandas', 'disk']:
            with self.subTest(engine=engine):
                with self.assertRaises((ValueError, HTTPException)):
                    self.ingest('group,amount\nA,USD 12\nB,€14\n', engine,
                        overrides=[ColumnOverride(column='amount', format='currency')])

    def test_late_decimal_cannot_round_an_earlier_exact_large_integer(self):
        source = 'group,amount\nA,9007199254740993\n' + 'A,1\n' * 49999 + 'B,0.5\n'
        for engine in ['pandas', 'disk']:
            with self.subTest(engine=engine):
                with self.assertRaises((ValueError, HTTPException)):
                    self.ingest(source, engine, overrides=[ColumnOverride(column='amount', parse_as='number')])
