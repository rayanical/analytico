"""Exact per-value equivalence of optimized full-column statistics."""
import itertools
import random
import unittest

import duckdb
import pandas as pd

from benchmarks.ingestion_last_three import legacy_stats_projection
from modules import disk_dataset as disk


class GuardedStatisticsTests(unittest.TestCase):
    def test_matches_legacy_for_numeric_and_unusual_text(self):
        rng = random.Random(20261007)
        values = [None, '', ' ', '\n', '001', '-0', '+001', '1.0', '.5',
                  '1.', '1e3', 'NaN', 'Infinity', '1,234', '1,234.5',
                  '0,001', '1%', ' 1 %', '$1', '$1,234.56', '€1.234,56',
                  '£1.25', '١٢٣', '１２３', '9007199254740993',
                  '18446744073709551616', '9' * 400]
        values += [''.join(s) for n in range(1, 4)
                   for s in itertools.product('01+.,$% ', repeat=n)]
        values += [''.join(rng.choices('019+-.,$% €£\t\nXYZ١',
                                     k=rng.randrange(1, 25)))
                   for _ in range(3000)]
        values += [symbol + tail for symbol in disk._CURRENCY_CHARS
                   for tail in ('1', '-1.25', '1,234.50', '001', 'foo', '1%')]
        dataset = disk.DiskDataset.__new__(disk.DiskDataset)
        dataset._raw_columns = ['c0']
        old, old_slots = legacy_stats_projection(dataset)
        new, new_slots = dataset._stats_projection()
        self.assertEqual(old_slots, new_slots)
        projections = [f'{p.rsplit(" AS ", 1)[0]} AS "{side}{i}"'
                       for side, group in [('o', old), ('g', new)]
                       for i, p in enumerate(group)]
        connection = duckdb.connect()
        try:
            connection.register('inputs', pd.DataFrame({
                'ordinal': range(len(values)), 'c0': values}))
            rows = connection.execute('SELECT ordinal,' + ','.join(projections)
                                      + ' FROM inputs GROUP BY ordinal ORDER BY ordinal').fetchall()
            width = len(old)
            for row in rows:
                self.assertEqual(row[1:1 + width], row[1 + width:],
                                 msg=f'Value index {row[0]}: {values[row[0]]!r}')
        finally:
            connection.close()
