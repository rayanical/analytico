# Import preview and reviewed schemas

Implemented priorities 1 and 2 from the production audit. No provider/model changes or dependency additions.

## User flow

1. Choose a CSV or demo. The backend saves an owned local copy and reads a bounded sample (256 KiB, at most 20 data records). This step does not call an AI provider or register a dataset.
2. Compare original and parsed values. Adjust delimiter, encoding, decimal/grouping separators, date order, and exact missing-value tokens. Recheck the saved sample before confirming changed settings.
3. Confirm. Validate quoting, record widths, encoding, NULs and limits across the entire file before ingestion. A good sample cannot conceal a malformed later record. Failure keeps the staged source available for corrections.
4. Review columns. Edit parsing, role, format, unit or aggregation; optionally copy a supported AI proposal into the form. Apply submits only changed fields. Reset removes explicit decisions and reruns automatic inference.
5. Applying edits rebuilds from the original CSV, validates the full column, and atomically replaces the dataset under the same public ID with a new version. Failed edits leave the active version intact; stale concurrent edits return 409. Existing read leases keep the retired source alive until their readers finish. Old chart history is disabled and dashboard snapshots for the replaced version are cleared.

The original and parsed preview tables show a sample, not a guarantee about every record. Full validation deliberately follows user confirmation so a wrongly guessed delimiter or locale can be corrected without rebuilding the entire dataset first.

## Parsing contract

Both ingestion engines use shared settings, structural validation, strict numeric validation and date format definitions. Forced conversions reject incompatible non-null values. Global number settings preserve likely identifiers and leading-zero lexemes unless the user explicitly requests a numeric conversion. Custom null tokens match exactly; values such as NA and NULL remain text by default.

Confirmed currency parsing checks symbols/codes against each other and against a specified unit. Percentage suffixes become ratios, while plain numeric ratios retain their values. Mixed percent suffixes/plain numbers and mixed currency identities are rejected. Numeric conversions preserve exact supported integers; conversions that would lose precision fail or retain text. Ambiguous forced dates require a compatible date order. Timezone-aware and submicrosecond timestamp lexemes remain text because the disk engine cannot preserve their full semantics in its current date representation.

Roles and aggregation recommendations feed profiling, default charts and the AI planner. AI proposals remain advisory until copied into editable fields and applied. Unsupported proposals remain visible but cannot be accepted as an inexpressible conversion.

## API and ownership

- `POST /imports/preview`: multipart CSV and optional settings JSON.
- `POST /imports/demo`: stage a bundled demo.
- `POST /imports/{id}/preview`: recheck settings.
- `POST /imports/{id}/confirm`: validate and register.
- `DELETE /imports/{id}`: release a staged import.
- `GET /datasets/{id}/schema`: settings, schema, version, bounded sample and proposals.
- `POST /datasets/{id}/schema`: expected version and partial column overrides.

Legacy direct upload/demo routes remain compatible and receive the same structural validation. Staging allows four imports, at most 512 MiB total, with lazy cleanup after 30 minutes of inactivity. Existing per-file, row, column and ingestion-concurrency limits remain enforced; headers are capped at 256 UTF-8 bytes and CSV fields at 262144 characters.

Source retention costs temporary local disk space, including on the pandas path. It enables reversible re-parsing without retaining another large in-memory copy. This is still session storage: restart persistence and durable workspaces are separate follow-ups. A bounded preview can also refuse a valid file when no complete data record fits inside its sample budget; it reports that limitation rather than guessing from a partial record.

## Validation

- 195 backend tests pass, including engine parity for locales, identifiers, duplicate headers, encodings, quoted newlines, null tokens, dates and reviewed currency/percent values; failed/stale edits, reset, source ownership, version changes and late malformed records.
- 16 offline AI evaluation tests pass. No live provider requests during validation.
- Frontend lint, helper checks, TypeScript and production build pass.
- Computer-use QA of the production build: demo staging → preview → confirmation → automatic column review; change population aggregation to mean; reopen and reset; verify profile changes and disabled old chart history. Screenshot: `backend/benchmarks/results/import-review-2026-10-02/column-review.jpg`.
- All nine benchmark chart payloads exactly match the previous committed baseline.

## Performance tradeoff

Three isolated worker trials per workload; medians, AI disabled. Timings cover local source parsing/ingestion, excluding browser upload transfer. Before is the preceding ingestion commit's recorded automatic-engine benchmark. These are synthetic workloads on this machine, not universal throughput guarantees.

| Workload | Before | With full validation | Before peak RSS | Current peak RSS |
| --- | ---: | ---: | ---: | ---: |
| 10,000 rows × 5 columns | 0.073 s | 0.075 s | 120.3 MiB | 120.5 MiB |
| 10,000 rows × 50 columns | 0.491 s | 0.607 s | 139.1 MiB | 137.8 MiB |
| 1,000,000 rows × 5 columns | 2.433 s | 3.118 s | 258.4 MiB | 259.6 MiB |

The added structural scan increases final ingestion latency, especially on wide files. It prevents implicit-index field loss and validates limits before pandas materialization. The million-row case remains about four times faster than the older 12.565-second baseline, with about 65% lower peak memory than its 745.5 MiB.

A separate million-row preview measurement took **76 ms median** for local source copying and bounded preflight (26,212,409-byte source, three warm-process trials, excludes browser transfer). It returns 20 rows; full validation waits for confirmation. Raw records are in `backend/benchmarks/results/import-review-2026-10-02/`.

Reproduce ingestion measurements from backend:

```sh
OPENAI_API_KEY='' COLUMN_INTERPRETER=off venv/bin/python benchmarks/ingestion.py \
  --engine auto --repeats 3 --output /private/tmp/import-review-auto.json
```

The test run also reports the existing Starlette/httpx deprecation and implicit temporary-directory cleanup warnings in legacy tests. Neither fails validation; the new ownership and replacement behavior is covered explicitly.
