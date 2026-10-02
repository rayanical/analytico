#!/usr/bin/env python3
"""Regenerate the offline snapshot from the current backend interpretation path."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
BACKEND = HERE.parent
DEFAULT_CASES = HERE / "interpretation_cases.json"
DEFAULT_OUTPUT = HERE / "baseline_current_repo.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    # The current cleaning path can optionally call a provider. Make the offline
    # behavior explicit before importing backend modules or loading any .env file.
    os.environ["OPENAI_API_KEY"] = ""
    os.environ["COLUMN_INTERPRETER"] = "off"
    sys.path.insert(0, str(BACKEND))
    try:
        import pandas as pd
    except ImportError:
        parser.error("snapshotting requires the backend Python dependencies, including pandas")

    from modules.data_janitor import clean_dataframe
    from modules.intelligence import auto_profile, detect_semantic_type

    cases = json.loads(args.cases.read_text(encoding="utf-8"))["cases"]
    predictions = []
    for case in cases:
        frame = pd.DataFrame({case["column_name"]: case["values"]})
        cleaned, actions, _, formats, semantic_suggestions = clean_dataframe(frame)
        column = cleaned.columns[0]
        role = semantic_suggestions.get(column) or detect_semantic_type(cleaned, column)
        profile = auto_profile(cleaned, {column: role}, formats)
        metric = next(
            (item for item in profile["top_metrics"] if item["name"] == column),
            None,
        )

        # These are the only decision fields exposed by the runtime path today.
        # In particular, do not infer unit, parsing policy, or clarification from
        # expected labels or from a generic display format.
        decision = {"role": role}
        if metric is not None:
            decision["recommended_aggregation"] = metric["aggregation"]
        predictions.append(
            {
                "case_id": case["id"],
                "decision": decision,
                "observed": {
                    "runtime_column_format": formats.get(column),
                    "profile_default_aggregation": metric["aggregation"] if metric else None,
                    "cleaning_actions": actions,
                },
            }
        )

    payload = {
        "model": "current_repo_offline",
        "provenance": {
            "python": sys.version.split()[0],
            "pandas": pd.__version__,
            "source_state": "working tree at snapshot time; no commit revision asserted",
            "provider_calls": "disabled by empty OPENAI_API_KEY",
        },
        "baseline_note": (
            "Offline snapshot from clean_dataframe, detect_semantic_type, and auto_profile. "
            "OPENAI_API_KEY was forced empty. Runtime format and cleaning actions are observations; "
            "unit, parsing policy, and clarification are unsupported and intentionally omitted."
        ),
        "predictions": predictions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(predictions)} offline predictions to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
