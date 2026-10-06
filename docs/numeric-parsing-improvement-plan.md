# Numeric Parsing Improvement Plan

> Historical proposal, not the current ingestion contract. Raw/parsed separation and loss-aware parsing are implemented. Partial-success coercion and separator guessing are not approved defaults: mixed or ambiguous values remain source text. Future interpretation changes should be evaluated against [the interpretation benchmark](interpretation-benchmark.md).

## Problem
Current ingestion can classify a column as a metric while values still contain textual units (for example `142 min`).
When downstream profiling expects pure numeric values, users may get hard-to-read conversion errors.

## Goals
- Parse numeric-like text robustly without requiring every alias to be hardcoded.
- Keep user-visible behavior safe and predictable.
- Avoid destructive conversion of original raw values.

## Proposed Approach
1. Keep raw + parsed representations
- Preserve original columns unchanged.
- Create parsed numeric views for candidate metric columns (for example internal `__numeric` series).

2. Add a deterministic number parser
- Support mixed separators (`1,234.56` and `1.234,56`) using rightmost-separator heuristic.
- Support magnitude suffixes (`k`, `m`, `b`) for values like `1.2m`.
- Normalize standard currency patterns before numeric conversion.

3. Add fuzzy unit normalization
- Use a small canonical unit set (minute, hour, day, etc.).
- Map noisy tokens (for example `mns`) to canonical units using fuzzy matching with confidence threshold.
- Convert to base units when confidence is high.

4. Apply safety gates
- Only coerce a column when parse success rate is high (for example >= 85%).
- If parse confidence is low, keep semantic type non-metric.
- Never silently overwrite source values.

5. AI fallback only for unresolved unit tokens
- Use AI only when deterministic + fuzzy logic cannot classify a unit token confidently.
- Cache token->unit decisions to avoid repeated calls and drift.

6. Improve observability
- Record parse strategy, success rate, and dropped-value count in cleaning actions.
- Surface concise warnings in the UI when metric parsing is partial.

## Immediate UX Safeguard
Short-term: return a concise, friendly ingestion error when a metric conversion failure occurs,
instead of exposing the full low-level Python exception text.
