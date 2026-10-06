# Opt-in column interpretation

Ingestion reuses bounded interpretation decisions keyed by the exact input, provider, model, prompt version, reasoning effort, and credential fingerprint. Every cache hit still undergoes full-column validation. The adapter and live evaluations bypass caching by default; see [cache limits and performance validation](performance-improvements-2026-10-02.md).

Column interpretation proposes a role, unit, parsing policy, aggregation, and whether clarification is needed. It cannot rename columns, execute code, impute observations, or authorize arbitrary calculations. The parsed view still requires deterministic evidence from the full column. The original frame remains stored separately.

## Local configuration

Keep keys in ignored `backend/.env`. Start with `backend/.env.example`; restart the backend after changes. Users supply their own provider keys.

- `COLUMN_INTERPRETER=off` is the default. Existing conservative parsing and keyless manual analytics continue.
- `COLUMN_INTERPRETER=luna` uses `OPENAI_API_KEY` and the Responses API. `COLUMN_INTERPRETER_MODEL` defaults to `gpt-6-luna`; outputs use a strict JSON schema, with response storage disabled.
- `COLUMN_INTERPRETER=jev` uses `AI_GATEWAY_API_KEY` and Vercel's evaluation endpoint, pinned to `typesafe-ai/jev` and the TypeSafe provider.

`OPENAI_MODEL` separately controls chart planning and business descriptions; it now defaults to `gpt-6-luna` too. These chat calls explicitly use no reasoning and bounded completion budgets. Setting interpretation to off does not disable those features or their outbound requests. Ingestion summaries still run automatically when an OpenAI key is configured. A per-dataset sharing control and key-management UI remain follow-ups.

## Bounds and runtime safety

Ingestion sends at most twelve evenly spaced values per column, including the tail, plus bounded column names and a small profile. Nulls are preserved in the sample. Adapter input is limited to 16 KiB; values and context have additional length/depth limits. The benchmark sends only its `column_name`, `values`, and `context`, never expected answers.

Ingestion attempts at most twelve columns, checking a twelve-second soft budget between calls. Each provider call has an eight-second timeout and no automatic retries; the last call can exceed the soft budget. Provider failure stops subsequent calls. Wide datasets and slow providers therefore produce skipped columns that need review.

An opted-in column with a failed, uncertain, skipped, contradictory, or unsupported decision retains its values and gets an unknown role. Automatic profiles and chart defaults omit it; AI chart plans referencing it return a clarification. Explicit manual analytics remain available with review warnings and existing operand checks.

Accepted metric parsing checks every non-null value. Leading-zero identifiers are protected. Currency parsing currently supports USD and EUR with explicit source evidence; a dollar symbol alone is insufficient. Ambiguous date order and mixed number conventions remain unparsed. Weighted rates, basis-point conversion, and last-by-entity aggregation are represented in the interpretation contract but are not runtime operations yet.

Numeric parsing additionally rejects loss of source precision and quantities or possible totals beyond JavaScript’s safe integer range. Large integer identifiers are serialized as exact strings for browser/filter round trips.

Upload column metadata exposes provider/model, prompt version, status, runtime acceptance, latency, reported token usage, sanitized failure code, and the proposed decision. Jev probabilities are separate from optional native confidence. They are not calibrated accuracy estimates. Background proposals are visible in the optional **Review columns** editor. They do not currently change the uploaded parsed view; supported proposals can be copied into the form and applied through full-source schema validation.

## Evaluation

See [the interpretation benchmark](interpretation-benchmark.md) for dry-run and explicitly opted-in live commands. The report records actual predictions, accuracy against fifteen synthetic cases, latency, usage, and provider failures. Cost stays unknown unless the provider reports it. This fixture is an initial regression set, not sufficient evidence to choose a production default.

See [the latest Luna improvement results](luna-improvement-results-2026-10-01.md) for the frozen prompt, repeated holdout outcomes, reasoning comparison, and remaining limits.
