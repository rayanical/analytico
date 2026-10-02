"""Production native CSV loader compatibility and fallback contracts."""
import io
import unittest
from unittest.mock import patch

from modules.disk_dataset import DiskDataset
from modules.import_policy import ImportSettings, CSVStructureError


class NativeCSVLoaderTests(unittest.TestCase):
    def create(self, source, settings=None, native=True):
        loader = DiskDataset._ingest_csv if native else DiskDataset._ingest_csv_chunks
        with patch.object(DiskDataset, '_ingest_csv', loader):
            dataset = DiskDataset.from_csv(io.BytesIO(source), import_settings=settings)
        self.addCleanup(dataset.close)
        return dataset

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

    def test_malformed_late_row_is_rejected_before_native_loading(self):
        source = ('code,amount\n'+'001,1\n'*21_000+'002,2,unexpected\n').encode()
        with self.assertRaises(CSVStructureError):
            self.create(source)

    def test_single_column_blank_lines_match_current_reader(self):
        source = b'label\nA\n\n   \nB\n'
        native = self.create(source)
        current = self.create(source, native=False)
        self.assertEqual(native.sample_frame(10).to_dict('list'), current.sample_frame(10).to_dict('list'))

    def test_long_valid_record_falls_back_without_truncation(self):
        values = ['a' * 200_000, 'b' * 200_000, 'c' * 200_000]
        source = ('first,second,third\n'+','.join(values)+'\n').encode()
        dataset = self.create(source)
        self.assertEqual(dataset.csv_loader, "pandas")
        self.assertEqual(dataset.csv_fallback_reason, 'native_reader_error')
        self.assertEqual(dataset.row_count, 1)
        self.assertEqual(dataset._connection.execute('SELECT c0,c1,c2 FROM source_data').fetchone(), tuple(values))
