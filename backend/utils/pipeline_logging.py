"""Structured ingestion timing without logging source data or credentials."""

from contextlib import contextmanager
import json
import sys
from time import perf_counter


def process_peak_rss_bytes():
    """Process lifetime high-water RSS, not an individual request's allocation."""
    try:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(peak if sys.platform == "darwin" else peak * 1024)
    except (ImportError, OSError, AttributeError):
        return None


class IngestionMeasurement:
    """Measure from CSV parse through response construction and dataset storage.

    Multipart transfer/parsing and FastAPI response serialization are outside this
    scope. RSS is shared across concurrent work; Windows reports null if the
    standard library cannot supply the high-water mark.
    """

    def __init__(self, endpoint):
        self.endpoint = endpoint
        self.phases = {}
        self.rows = None
        self.columns = None

    def __enter__(self):
        self.started = perf_counter()
        self.start_peak = process_peak_rss_bytes()
        return self

    @contextmanager
    def phase(self, name):
        start = perf_counter()
        try:
            yield
        finally:
            self.phases[name] = self.phases.get(name, 0.0) + perf_counter() - start

    def __exit__(self, error_type, error, traceback):
        record = {
            "event": "ingestion_performance", "endpoint": self.endpoint,
            "status": "error" if error_type else "ok",
            "total_seconds": perf_counter() - self.started,
            "phase_seconds": self.phases, "rows": self.rows, "columns": self.columns,
            "process_peak_rss_start_bytes": self.start_peak,
            "process_peak_rss_bytes": process_peak_rss_bytes(),
            "memory_scope": "process_lifetime_high_water",
            "timing_scope": (
                "handler_parse_through_storage_excludes_transfer_and_serialization"
                if "csv_parse" in self.phases else
                "dataframe_ingestion_through_storage_excludes_csv_parse_transfer_and_serialization"
            ),
        }
        print(json.dumps(record, allow_nan=False))
        return False
