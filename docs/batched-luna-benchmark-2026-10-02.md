# Luna column batching: quick experiment (2026-10-02)

Live model: `gpt-6-luna`, reasoning `none`. Existing interpretation contract unchanged. No production batching or automatic schema mutation was enabled.

| Columns/workload | Current four concurrent column requests | One whole-schema request (two runs) | Eight-column batches, four concurrent requests (two runs) |
| --- | ---: | ---: | ---: |
| 30 regression cases | 15.03 s | 9.82–10.88 s | 3.31–3.51 s |
| 20 taxi columns | 5.58 s | 7.08–7.30 s | 3.21–3.29 s |
| 50 synthetic columns | 14.16 s | 14.65–15.66 s | 5.94–6.09 s |

These timings measure column inference only, excluding ingestion, upload, business summary, chart generation and rendering. Current requests use their existing eight-second timeout; experimental batched requests use 25 seconds. The current baseline has one run; each experimental arm has two. Whole-schema second runs shuffle column order; chunked repeats retain order. Provider caching and transient latency were not controlled. Results are directional, not production latency guarantees.

On the existing 30-case regression set, every arm assigned all 30 roles correctly and missed zero required clarifications. Complete five-field matches were 27/30 currently, 28/30 in each whole-schema run, and 26/30 in each chunked run. This previously used regression set is not an unseen accuracy evaluation. Taxi and synthetic schemas have no complete gold labels; agreement is not proof of correctness. Taxi rate-code interpretation changed with column order in the whole-schema arm.

Recommendation: do not promote one giant request. Bounded parallel batches are promising for a future advisory implementation, but their small complete-decision accuracy regression needs investigation before replacing the current adapter. Do not automatically apply model conversions based on this experiment. The upload toggle uses the established interpreter and optional Review data workflow.

The harness rejects missing, duplicate, or invented column indices and uses structured output. Valid output shape does not establish semantic correctness ([OpenAI structured-output documentation](https://developers.openai.com/api/docs/guides/structured-outputs)). Existing full-column validation remains necessary when applying any proposed parsing change.

Artifacts: `backend/benchmarks/results/batched-luna-2026-10-02/results.json` and `chunked.json`. Reproduce with `backend/venv/bin/python backend/benchmarks/batched_luna.py --live --output <file>`; add `--chunked-only` for the bounded-batch arm. Requires a configured API key and sends bounded samples to the provider.

Cleanup: the rejected batching runner is archived in `/private/tmp/analytico-retired-experiments-2026-10-03.tar.gz`, and raw responses in `/private/tmp/analytico-benchmark-archive-2026-10-03.tar.gz`. Restore these before following the historical reproduction instructions above.
