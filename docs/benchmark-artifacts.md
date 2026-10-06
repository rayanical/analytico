# Benchmark artifact policy

Keep compact comparison/summary results, regression fixtures, source manifests and reproducible runners. Raw responses, repeated per-dataset trials, sampled-row snapshots and screenshots are generated diagnostics, not runtime data.

The October 3 cleanup removed 103 raw files, 176,769 text lines and 5,074,065 bytes from `backend/benchmarks/results`. It retained the canonical deterministic and performance before/after evidence, compact summaries, library environment freezes, application files, and all tests. Public real datasets were used for the large-file benchmarks; tiny synthetic safety fixtures in tests exercise specific edge cases and are intentionally retained.

Removed diagnostics are recoverable from `/private/tmp/analytico-benchmark-archive-2026-10-03.tar.gz`. The list is in `/private/tmp/analytico-benchmark-cleanup-manifest.json`. These temporary files may be cleared by the operating system. Restore archived results before rescoring those historical runs, or regenerate them with the runners. Avoid checking raw generated samples and model responses into source control; `.gitignore` now excludes them. Preserve compact measured summaries rather than deleting evidence of regressions or unmet targets.

Retired 6 one-off probe/library/batching runners (1,479 source lines) into `/private/tmp/analytico-retired-experiments-2026-10-03.tar.gz`. These rejected experiments are outside the runtime and have no test imports. Keep active Luna and real-dataset acceptance runners. Historical report reproduction requires restoring the retired runners first.
