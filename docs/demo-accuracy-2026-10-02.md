# Demo accuracy follow-up

The taxi measurement-as-identifier issue was reproduced and minimized to a generic
numeric column mixing plain decimals with grouped values such as `1,243.5` and
`2,109`. The previous parser required uniform formatting, retained the entire
column as text and classified its high-cardinality text as identifiers.

Both readers now accept a complete column of valid plain or three-digit-grouped
numbers when decimal-point evidence establishes the consistent notation. No taxi
column names or dataset-specific mappings enter this decision. Grouped-only
ambiguous values, malformed grouping, leading-zero identifiers and large integers
that would lose precision remain source text. The assumption is that a column
uses consistent locale semantics; contradictory business metadata still requires
an explicit reviewed policy. This does not solve arbitrary mixed locales.

Planner guidance now requires clarification when a business term has multiple
plausible measures, and prohibits invented weights, denominators, currency
identities or business definitions. This is a prompt safeguard, not deterministic
proof that every ambiguous request will be caught. OpenAI documents that
[structured outputs can still contain semantic mistakes](https://developers.openai.com/api/docs/guides/structured-outputs).

Validation: 214 backend tests passed. Two generic regression tests compare both
engines' expected values and preservation behavior. The original failing fixture
now passes. A full default-path taxi import retained 1,068,755 rows, used the native
loader and classified trip distance, fare amount and total amount as metrics.
Ingestion took 15.732 seconds in this single run.

Ten live GPT-6 Luna checks passed: two repetitions each for average distance, total
fares, total tips, ambiguous revenue and unsupported weighted average. Explicit
requests returned the requested axes and aggregation; the latter two returned
clarification. Question responses took 2.77–4.79 seconds in this run. This checks
planning behavior, not labeled accuracy over every possible dataset or question.
Results are in `backend/benchmarks/results/demo-accuracy-2026-10-02/smoke.json`.

Ordinary decimal analytics remain floating point. Exact accounting arithmetic and
automatic discovery of missing business definitions were not added for the demo.
Use explicit measure names for a predictable demonstration.

Batched Luna schema proposals remain the next experiment after the demo: one
bounded request for ordinary schemas, bounded batches for very wide files,
independent parsing/role/unit/aggregation fields, full-column conversion validation,
and optional editing. Successful numeric conversion cannot by itself prove metric
meaning. No batched or automatic AI application was enabled in this follow-up.
