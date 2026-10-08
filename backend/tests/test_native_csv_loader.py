"""Production native CSV loader compatibility and fallback contracts."""
import io
import os
import unittest
from unittest.mock import patch

from modules.disk_dataset import DiskDataset
from modules.import_policy import ImportSettings, CSVStructureError, validate_csv_structure


class NativeCSVLoaderTests(unittest.TestCase):
    def create(self, source, settings=None, native=True):
        loader = DiskDataset._ingest_csv if native else DiskDataset._ingest_csv_chunks
        with patch.object(DiskDataset, '_ingest_csv', loader):
            dataset = DiskDataset.from_csv(io.BytesIO(source), import_settings=settings)
        self.addCleanup(dataset.close)
        return dataset

    def outcome(self, source, settings=None, *, native_validation='0', **kwargs):
        try:
            with patch.dict(os.environ, {'ANALYTICO_NATIVE_VALIDATION': native_validation}):
                dataset = DiskDataset.from_csv(
                    io.BytesIO(source), import_settings=settings, **kwargs,
                )
            self.addCleanup(dataset.close)
            headers, rows, row_count = dataset.raw_source_snapshot(
                limit=1_000, evenly_spaced=False,
            )
            return ('accepted', headers, rows, row_count)
        except Exception as error:
            return ('rejected', type(error))

    def test_lexemes_dialects_nulls_and_source_are_preserved(self):
        cases = [
            (b'code;amount;label\n001;"1.234,56";"a;\nb"\n002;NULL;NA\n003;;\n',
             ImportSettings(delimiter=';', decimal_separator=',', grouping_separator='.', null_values=['NULL']),
             [('001', '1.234,56', 'a;\nb'), ('002', None, 'NA'), ('003', '', '')]),
            (b'code,label\n001,A\n\n002,B\n', None, [('001', 'A'), ('002', 'B')]),
            (b'code,amount\n001,9007199254740993\n002,""\n', ImportSettings(null_values=[]),
             [('001', '9007199254740993'), ('002', '')]),
            ('code,label\n001,caf\u00e9\n'.encode('cp1252'), ImportSettings(encoding='cp1252'), [('001', 'caf\u00e9')]),
            ('code,label\n001,caf\u00e9\n'.encode('utf-16'), ImportSettings(encoding='utf-16'), [('001', 'caf\u00e9')]),
        ]
        for source, settings, expected in cases:
            with self.subTest(settings=settings):
                native = self.create(source, settings)
                current = self.create(source, settings, native=False)
                self.assertEqual(native.source_path.read_bytes(), source)
                raw = ', '.join(native._raw_columns)
                values = native._connection.execute(f'SELECT {raw} FROM source_data ORDER BY _row_ordinal').fetchall()
                self.assertEqual(values, expected)
                self.assertEqual(native.column_schema, current.column_schema)
                self.assertEqual(native.sample_frame(10).to_dict('list'), current.sample_frame(10).to_dict('list'))

    def test_parallel_reader_keeps_all_rows_in_original_order(self):
        source = ('code,amount\n'+''.join(f'{i:08d},{i}\n' for i in range(150_000))).encode()
        dataset = self.create(source)
        incorrect = dataset._connection.execute('SELECT count(*) FROM source_data WHERE c0 != printf(\'%08d\', _row_ordinal) OR c1 != CAST(_row_ordinal AS VARCHAR)').fetchone()[0]
        self.assertEqual(dataset.csv_loader, "native")
        self.assertEqual(dataset.row_count, 150_000)
        self.assertEqual(incorrect, 0)

    def test_mixed_grouped_and_plain_decimals_parse_in_both_engines(self):
        from modules.data_janitor import clean_dataframe
        from utils.dataframe_utils import read_csv_fast
        source = ('reading\n"1,243.5"\n"2,109"\n'+''.join(f'{i/10}\n' for i in range(100))).encode()
        disk = self.create(source)
        self.assertEqual(disk.column_types['reading'], 'metric')
        values = disk.sample_frame(200)['reading'].tolist()
        self.assertEqual(values[:2], [1243.5, 2109.0])
        cleaned, *_ = clean_dataframe(read_csv_fast(io.BytesIO(source)), interpret_columns=False)
        self.assertEqual(cleaned['reading'].tolist(), values)

    def test_ambiguous_or_malformed_grouping_remains_source_text(self):
        from modules.data_janitor import clean_dataframe
        from utils.dataframe_utils import read_csv_fast
        for values in [['1,234', '2,345'], ['1,23', '2.5'], ['001', '2.5'], ['9,007,199,254,740,993', '2.5']]:
            with self.subTest(values=values):
                source = ('reading\n'+''.join('"'+v+'"\n' for v in values)).encode()
                disk = self.create(source)
                self.assertEqual(disk.sample_frame(10)['reading'].tolist(), values)
                cleaned, *_ = clean_dataframe(read_csv_fast(io.BytesIO(source)), interpret_columns=False)
                self.assertEqual(cleaned['reading'].tolist(), values)

    def test_malformed_late_row_is_rejected_before_native_loading(self):
        source = ('code,amount\n'+'001,1\n'*21_000+'002,2,unexpected\n').encode()
        with self.assertRaises(CSVStructureError):
            self.create(source)

    def test_single_column_blank_lines_match_current_reader(self):
        source = b'label\nA\n\n   \nB\n'
        native = self.create(source)
        current = self.create(source, native=False)
        self.assertEqual(native.sample_frame(10).to_dict('list'), current.sample_frame(10).to_dict('list'))

    def test_leading_blank_physical_record_is_skipped_with_and_without_bom(self):
        sources = [
            b'\ncode,label\n001,A\n',
            b'\xef\xbb\xbf\ncode,label\n001,A\n',
        ]
        for source in sources:
            for native_validation in ('0', '1'):
                with self.subTest(bom=source.startswith(b'\xef\xbb\xbf'),
                                  native_validation=native_validation):
                    outcome = self.outcome(source, native_validation=native_validation)
                    self.assertEqual(
                        outcome,
                        ('accepted', ['code', 'label'], [['001', 'A']], 1),
                    )

    def test_long_valid_record_falls_back_without_truncation(self):
        values = ['a' * 200_000, 'b' * 200_000, 'c' * 200_000]
        source = ('first,second,third\n'+','.join(values)+'\n').encode()
        dataset = self.create(source)
        self.assertEqual(dataset.csv_loader, "pandas")
        self.assertEqual(dataset.csv_fallback_reason, 'native_reader_error')
        self.assertEqual(dataset.row_count, 1)
        self.assertEqual(dataset._connection.execute('SELECT c0,c1,c2 FROM source_data').fetchone(), tuple(values))
        self.assertEqual(len(dataset.raw_source_snapshot(limit=1, evenly_spaced=False)[1][0][0]), 160)

    def test_native_validation_experiment_matches_existing_parser_contract(self):
        ascii_boundary = 'a' * (256 * 1024)
        ascii_over_limit = 'a' * (256 * 1024 + 1)
        unicode_boundary = 'é' * (256 * 1024)
        unicode_over_limit = 'é' * (256 * 1024 + 1)
        valid_headers = ','.join(['a'] * 256).encode() + b'\n' + b','.join([b'x'] * 256) + b'\n'
        too_many_columns = ','.join(['a'] * 257).encode() + b'\n' + b','.join([b'x'] * 257) + b'\n'
        cases = [
            (b'id;note\n001;"first line\nsecond line"\n002;NIL\n003;"NIL"\n',
             ImportSettings(delimiter=';', null_values=['NIL']), {}),
            (b'code,label\n001,\n002,NULL\n', ImportSettings(null_values=[]), {}),
            (b'code,label\n\n001,A\n', None, {}),
            (b'code,label\n  \n001,A\n', None, {}),
            (b'code,label\n\t\n001,A\n', None, {}),
            (b'a,b,c\n1,"x" ,2\n', None, {}),
            (b'a,b,c\n1,"x"\t,2\n', None, {}),
            (b'a,b\n1,"x" \n', None, {}),
            (b'a,b\n1,"x"\t\n', None, {}),
            (b'a,b\n1,"x" ', None, {}),
            (b'code,label\n""," "\n', None, {}),
            (b'code,label\n""," "\n', ImportSettings(null_values=[]), {}),
            (b'code,label\n"   "\n', None, {}),
            (b'code,label\n001,"open\n', None, {}),
            (b'code,label\n001,"closed"tail\n', None, {}),
            (b'code,label\n001\n', None, {}),
            (b'code,label\n001,A,extra\n', None, {}),
            (b'code,label\n001,has\x00nul\n', None, {}),
            (b'code,la\x00bel\n001,A\n', None, {}),
            (('code,value\n001,' + ascii_boundary + '\n').encode(), None, {}),
            (('code,value\n001,' + ascii_over_limit + '\n').encode(), None, {}),
            (('code,value\n001,' + unicode_boundary + '\n').encode(), None, {}),
            (('code,value\n001,' + unicode_over_limit + '\n').encode(), None, {}),
            (('a' * 256 + ',b\nx,y\n').encode(), None, {}),
            (('a' * 257 + ',b\nx,y\n').encode(), None, {}),
            (valid_headers, None, {}),
            (too_many_columns, None, {}),
            (b'code,label\n001,A\n002,B\n', None, {'max_rows': 1}),
            (b'label\n\nA\n   \nB\n', None, {}),
            ('code,label\n001,café\n'.encode('cp1252'), ImportSettings(encoding='cp1252'), {}),
            ('code,label\n001,café\n'.encode('utf-16'), ImportSettings(encoding='utf-16'), {}),
            (b'code,label\n001,\xff\n', None, {}),
        ]
        for source, settings, kwargs in cases:
            with self.subTest(settings=settings, size=len(source)):
                self.assertEqual(
                    self.outcome(source, settings, native_validation='1', **kwargs),
                    self.outcome(source, settings, native_validation='0', **kwargs),
                )

    def test_native_validation_skips_python_full_scan_only_for_native_loads(self):
        source = b'code,label\n001,A\n002,B\n'
        with patch.dict(os.environ, {'ANALYTICO_NATIVE_VALIDATION': '1'}):
            with patch('modules.disk_dataset.validate_csv_structure', wraps=validate_csv_structure) as validator:
                dataset = DiskDataset.from_csv(io.BytesIO(source))
        self.addCleanup(dataset.close)
        self.assertEqual(dataset.csv_loader, 'native')
        validator.assert_not_called()

        quoted = b'code,label\n001,"A"\n'
        with patch.dict(os.environ, {'ANALYTICO_NATIVE_VALIDATION': '1'}):
            with patch('modules.disk_dataset.validate_csv_structure', wraps=validate_csv_structure) as validator:
                quoted_dataset = DiskDataset.from_csv(io.BytesIO(quoted))
        self.addCleanup(quoted_dataset.close)
        validator.assert_called_once()

        malformed = b'code,label\n001,A\n002,B,extra\n'
        with patch.dict(os.environ, {'ANALYTICO_NATIVE_VALIDATION': '1'}):
            with patch.object(DiskDataset, '_ingest_csv', DiskDataset._ingest_csv_chunks):
                with self.assertRaises(CSVStructureError):
                    DiskDataset.from_csv(io.BytesIO(malformed))

    def test_source_loaded_callback_is_bounded_early_and_failure_is_optional(self):
        source = ('code,label\n' + ''.join(
            f'{index:03d},{"" if index == 0 else f"row-{index}"}\n'
            for index in range(25)
        )).encode()
        observed = {}

        def callback(dataset):
            observed['summaries_at_callback'] = len(dataset.column_summaries)
            observed['snapshot'] = dataset.raw_source_snapshot(limit=12)
            observed['first_rows'] = dataset.raw_source_snapshot(limit=3, evenly_spaced=False)
            raise RuntimeError('private callback detail')

        with self.assertLogs('modules.disk_dataset', level='WARNING') as messages:
            dataset = DiskDataset.from_csv(io.BytesIO(source), on_source_loaded=callback)
        self.addCleanup(dataset.close)
        self.assertEqual(observed['summaries_at_callback'], 0)
        headers, rows, row_count = observed['snapshot']
        expected_positions = sorted({round(i * (24) / 11) for i in range(12)})
        self.assertEqual(headers, ['code', 'label'])
        self.assertEqual([row[0] for row in rows], [f'{position:03d}' for position in expected_positions])
        self.assertEqual(row_count, 25)
        self.assertEqual(observed['first_rows'][1], [
            ['000', None], ['001', 'row-1'], ['002', 'row-2'],
        ])
        self.assertEqual(dataset.row_count, 25)
        self.assertNotIn('private callback detail', '\n'.join(messages.output))

        callback_calls = []
        with patch.dict(os.environ, {'ANALYTICO_NATIVE_VALIDATION': '1'}):
            with self.assertRaises(CSVStructureError):
                DiskDataset.from_csv(
                    io.BytesIO(b'code,label\n001,A\n002,B,extra\n'),
                    on_source_loaded=lambda _dataset: callback_calls.append(True),
                )
        self.assertEqual(callback_calls, [])
