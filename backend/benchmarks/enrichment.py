"""Offline first-response benchmark with a simulated one-second AI request."""
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import core.config  # noqa: E402,F401
from services.ingestion_service import ingest_dataframe
from services.enrichment_service import get_enrichment_status
from storage import DATASETS
import pandas as pd


def run(deferred):
    frame = pd.DataFrame({"category": ["a", "b"] * 5000, "amount": ["10", "20"] * 5000})
    def summary(*args):
        time.sleep(1)
        return "Simulated summary"
    with patch.dict(os.environ, {"OPENAI_API_KEY": "offline-mock", "COLUMN_INTERPRETER": "off"}), \
         patch("services.ingestion_service._generate_business_summary", side_effect=summary), \
         contextlib.redirect_stdout(io.StringIO()):
        start = time.perf_counter()
        response = ingest_dataframe(frame, "synthetic.csv", "/benchmark", defer_enrichment=deferred)
        first = time.perf_counter() - start
        if deferred:
            deadline = start + 10
            while True:
                status = get_enrichment_status(response.dataset_id)
                if status["status"] in {"done", "error", "disabled"}:
                    break
                if time.perf_counter() > deadline:
                    raise AssertionError("Enrichment did not complete")
                time.sleep(.01)
            assert status["status"] == "done" and status["summary"] == "Simulated summary"
        else:
            assert response.summary == "Simulated summary"
        complete = time.perf_counter() - start
        dataset = DATASETS.pop(response.dataset_id)
        assert dataset.df["amount"].sum() == 150000
        dataset.close()
        return {"deferred": deferred, "first_response_seconds": first, "enrichment_complete_seconds": complete}


if __name__ == "__main__":
    results = [run(deferred) for deferred in [False, True] for _ in range(3)]
    print(json.dumps({"simulated_ai_delay_seconds": 1, "live_provider_calls": 0, "results": results}, indent=2))
