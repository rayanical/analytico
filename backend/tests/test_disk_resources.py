"""Local worker budgets must respect host capacity and explicit settings."""
import io
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modules.disk_dataset import DiskDataset, _duckdb_thread_count


class DiskResourceTests(unittest.TestCase):
    def test_default_respects_cpu_capacity_and_unknown_capacity(self):
        with patch.dict(os.environ, {}, clear=True):
            for capacity, expected in [(8, 4), (2, 2), (1, 1), (None, 2)]:
                with self.subTest(capacity=capacity), patch('modules.disk_dataset.os.cpu_count', return_value=capacity):
                    self.assertEqual(_duckdb_thread_count(), expected)

    def test_explicit_budget_is_applied_to_real_connection(self):
        with patch.dict(os.environ, {'ANALYTICO_DUCKDB_THREADS': '1'}), patch('modules.disk_dataset.os.cpu_count', return_value=8):
            dataset = DiskDataset.from_csv(io.BytesIO(b'amount\n10\n20\n'), filename='budget.csv')
            self.addCleanup(dataset.close)
            self.assertEqual(dataset._connection.execute("SELECT current_setting('threads')").fetchone()[0], 1)
        with patch.dict(os.environ, {'ANALYTICO_DUCKDB_THREADS': '8'}), patch('modules.disk_dataset.os.cpu_count', return_value=2):
            self.assertEqual(_duckdb_thread_count(), 2)

    def test_invalid_budget_fails_as_configuration_error(self):
        for value in ['0', '-1', '9', 'garbage', '1.5', '']:
            with self.subTest(value=value), patch.dict(os.environ, {'ANALYTICO_DUCKDB_THREADS': value}):
                with self.assertRaisesRegex(RuntimeError, 'ANALYTICO_DUCKDB_THREADS'):
                    _duckdb_thread_count()
