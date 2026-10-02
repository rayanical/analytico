"""Experimental native CSV bridge; never imported by production services.

Keep full structural validation, source retention, conversion and profiling.
Only replace pandas-to-DuckDB chunk transport for UTF-8 sources.
"""
import duckdb
import pandas as pd
from modules.disk_dataset import DiskDataset, _q
from modules.import_policy import reader_options

ORIGINAL_LOADER = DiskDataset._ingest_csv_chunks


def native_loader(self, requested_chunk_size):
    self._benchmark_native_used = False
    if self.import_settings.encoding not in {'utf-8', 'utf-8-sig'}:
        self._benchmark_native_fallback = 'encoding'
        return ORIGINAL_LOADER(self, requested_chunk_size)
    headers = pd.read_csv(self.source_path, nrows=0, **reader_options(self.import_settings)).columns.tolist()
    # Single-column blank physical lines have different native-reader semantics.
    # Retain the original reader for this class instead of filtering decoded
    # values, which would incorrectly discard quoted empty/whitespace records.
    if len(headers) == 1:
        self._benchmark_native_fallback = 'single_column_blank_lines'
        return ORIGINAL_LOADER(self, requested_chunk_size)
    self._initialize_source_table(headers)
    columns = ', '.join(_q(column) for column in self._raw_columns)
    # Parameterize all source data/options; only generated c0... column names
    # enter SQL. No AI-generated SQL or type inference is involved.
    try:
        self._connection.execute(
            f'INSERT INTO source_data SELECT row_number() OVER () - 1, {columns} '
            'FROM read_csv(?, columns=?, header=true, auto_detect=false, '
            'delim=?, nullstr=?, force_not_null=?, encoding=\'utf-8\', '
            'quote=\'"\', escape=\'"\', parallel=true, strict_mode=true, '
            'ignore_errors=false, null_padding=false, max_line_size=524288, buffer_size=8388608)',
            [str(self.source_path), {column: 'VARCHAR' for column in self._raw_columns},
             self.import_settings.delimiter, self.import_settings.null_values or [''],
             self._raw_columns if not self.import_settings.null_values else []],
        )
    except duckdb.Error:
        # Bounded native buffers may reject otherwise supported wide/long rows.
        # Discard partial raw data and rebuild with the established reader.
        self._connection.execute('DROP TABLE source_data')
        self.row_count = 0
        self._benchmark_native_fallback = 'native_reader_error'
        return ORIGINAL_LOADER(self, requested_chunk_size)
    self._benchmark_native_used = True
    self.row_count = self._connection.execute('SELECT count(*) FROM source_data').fetchone()[0]
    if self.row_count > self._max_rows:
        raise ValueError('CSV exceeds the configured row limit.')
