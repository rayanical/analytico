# Production next steps — October 2, 2026

Audit of commit `fead9e1`, after the ingestion and memory improvements. Target: a downloaded, single-user, local analytics app with optional GPT-6 Luna and user-supplied credentials. This is a current code and product review, not a claim of exhaustive security certification. Implementation was left unchanged. Three Luna subagents inspected ingestion/storage, AI/analytics, and frontend/release concerns; findings below were checked against current source. Existing passing checks and measurements are recorded separately in [the ingestion report](ingestion-performance-2026-10-02.md).

## What to retain

Keep conservative source preservation, null preservation, closed parameterized analytics, model-output/schema validation, invalid-filter rejection, bounded responses, resource caps, and the prohibition on generated Python execution. The recent performance changes apply to general data structures. They do not establish identical parsing behavior for every format, or correct business semantics for every column.

If “rejects” means regex: retain syntax checks for numeric/date representation and identifiers. Replace business-meaning guesses progressively with an explicit, editable schema. If it means rejected inputs or requests: distinguish invalid structure, unsafe conversion, unsupported operation, and unresolved interpretation. Uncertain columns should remain available as source text; a rejected operation should explain how the user can resolve it. Do not turn an unsafe or unsupported operation into an accepted but incorrect result.

## Ranked work

### 1. Import preflight and explicit format settings

**Payoff: highest; correctness before more intelligence. Scope: medium.**

Both readers assume pandas' default comma-separated format: [small reader](../backend/utils/dataframe_utils.py#L18), [disk reader](../backend/modules/disk_dataset.py#L308). There is no import setting for delimiter, encoding, decimal/group separator, date order, or missing-value tokens. The current missing-token policy intentionally preserves literal `NA` and `NULL`; expanding that policy silently would be a regression.

Reproduced with `name;amount\nAlice;1234,56\nBob;5678,90\n`: the small reader returns one column, `name;amount`, values `56`/`90`, and implicit index labels `Alice;1234`/`Bob;5678`. Those index fields are not analytics columns. A tab-separated input becomes one combined column. Checking the resulting DataFrame width does not detect this source-shape problem; the disk insertion checks parsed width at [disk_dataset.py:345](../backend/modules/disk_dataset.py#L345).

Add bounded preflight that proposes dialect/settings and shows an original-versus-parsed preview. Validate source row widths and prevent implicit-index inference from silently removing fields. Detect uncertainty rather than treating every one-column file as malformed: legitimate one-column CSVs must still work. Check the full stream for late structural errors. For malformed rows, provide source row locations and a choice to correct or explicitly quarantine them; never silently skip them.

Completion: comma, semicolon, tab, quoted separators/newlines, BOM/encoding variants, duplicate headers, short/long rows and legitimate one-column files have defined behavior. Preview and final parsing use the same settings. Source fields cannot disappear without an explicit recorded decision.

### 2. One parsing policy and a reviewed column schema

**Payoff: highest for generality. Scope: medium to large.**

The file-size threshold selects an execution engine, but can also change meaning. Synthetic checks confirmed that pandas accepts offset timestamps and nanosecond UTC timestamps while disk preserves them as text. See [pandas date parsing](../backend/modules/data_janitor.py#L418) and [disk precision preservation](../backend/modules/disk_dataset.py#L629). Both preserve source values, but time analytics availability differs for the same input. Locale numbers such as `1.234,56` remain text without a way to specify the intended convention.

Normal uploads clean with interpretation disabled and return background proposals separately: [ingestion_service.py:67](../backend/services/ingestion_service.py#L67), [proposal worker](../backend/services/ingestion_service.py#L120). There is no ordinary review/apply workflow. The test proving applied interpretation calls the direct ingestion helper, not the normal upload route. Pandas and disk also duplicate semantic rules, including the `life_exp` literal and non-additive keyword lists: [intelligence.py:146](../backend/modules/intelligence.py#L146), [disk_dataset.py:802](../backend/modules/disk_dataset.py#L802).

Create a small shared parsing-policy interface consumed by both engines. Represent column identity, original name, role, units/currency, decimal/date conventions, null policy, aggregation recommendation, and decision provenance explicitly. Luna proposes; the user can accept or edit; deterministic full-column validation applies supported changes to a versioned parsed view. Manual overrides must work without an AI key. Preserve original data and permit undo. Advance dataset/cache versions and refresh affected charts after application.

Tradeoff: shared policy reduces drift without forcing one engine for every file. Some engine capabilities differ, so represent unsupported capabilities explicitly instead of obscuring them. Avoid a general transformation language in the first iteration; start with the parsing and metadata edits the current app can validate.

Completion: forced pandas/disk imports have equivalent agreed semantics for supported inputs; unsupported timezone/resolution cases are consistent and explained. Accepted settings are persisted and actually influence analytics. Renaming a measure does not silently redefine its aggregation meaning.

### 3. Make counting and supported analytics unambiguous

**Payoff: high; comparatively bounded change. Scope: small to medium.**

Runtime `count` counts non-null measure observations: [aggregation.py:128](../backend/modules/aggregation.py#L128), [disk aggregation](../backend/modules/disk_dataset.py#L1188). The query prompt correctly distinguishes this from distinct entities at [config.py:82](../backend/core/config.py#L82), but interpretation describes count as rows or distinct identifiers at [column_interpretation.py:149](../backend/modules/column_interpretation.py#L149). The holdout fixture asks for distinct machines and expects unqualified count at [interpretation_holdout.json:28](../backend/evals/interpretation_holdout.json#L28). Those contracts should agree.

Add explicit record count, non-null value count, and distinct-value count with documented null behavior. Keep existing count backward compatible or migrate it explicitly; do not silently change saved chart meaning. Update the planner, UI, both engines, labels, and eval expectations together.

The interpretation vocabulary includes `ratio_of_sums` and `last_by_entity`, but current application validation rejects them at [data_janitor.py:490](../backend/modules/data_janitor.py#L490). Retain that rejection until the executor supports them. Distinguish “unsupported” from “uncertain” in proposals and evaluation. Weighted rates and stock/snapshot operations can follow the reviewed schema; they require numerator/denominator or entity/time definitions, not just more keyword matching.

Completion: repeated IDs and missing measures produce independently verified row/value/distinct counts, identically across engines. Unsupported plans cannot be presented as applied calculations.

### 4. Durable local workspaces and reproducible provenance

**Payoff: high for a downloaded product. Scope: large.**

The backend registry is process-local, capped at ten, and expires idle data: [storage.py:130](../backend/storage.py#L130), [expiry](../backend/storage.py#L83). Disk data is temporary and removed on close: [disk_dataset.py:1428](../backend/modules/disk_dataset.py#L1428). Browser charts persist while their source datasets cannot be reopened after restart. Multiple backend workers would also have separate registries; the current app should remain a single backend process until that contract changes.

Store local workspace manifests, stable dataset identity/version, source hash/settings, original source bytes, parsed storage, chart definitions and review decisions. Commit imports atomically; detect interrupted imports; provide explicit close, delete and export/backup controls. Preserve raw bytes for small inputs too: their current `raw_df` is a copy after CSV parsing at [ingestion_service.py:64](../backend/services/ingestion_service.py#L64), not the original file.

Browser persistence is not a reliable backup: [writeStorage](../frontend/src/context/DataContext.tsx#L99) logs storage failures without telling the user. “Upload New” clears active state, not associated snapshots: [clearData](../frontend/src/context/DataContext.tsx#L625). Add explicit per-workspace deletion and visible save failures. Use session eviction to release open handles, not to destroy saved workspaces.

Tradeoff: persistent files require retention, version migration and recovery rules. Start with a local manifest plus existing disk storage rather than introducing a hosted database or account system.

Completion: restart/reopen, interrupted import, schema-version change, disk-full/save failure, delete, and backup/restore flows retain or deliberately remove the correct data and charts.

### 5. Explicit AI sharing and usable BYOK settings

**Payoff: high for trust and onboarding. Scope: medium.**

With a configured OpenAI key, upload queues a summary even when column interpretation is off: [ingestion_service.py:127](../backend/services/ingestion_service.py#L127). It sends filename, up to three rows and twenty columns: [summary generation](../backend/services/ingestion_service.py#L23). Query planning sends sample rows on explicit AI queries as expected, but no per-dataset sharing policy governs these features.

Provide local credential/model settings and a visible per-dataset sharing choice covering summaries, interpretation, planning and analysis. Make keyless behavior clear. Enforce the policy in the backend, not only the UI. Allow schema-only context where practical; show which context is sent. Keep GPT-6 Luna as the unified default. Avoid browser storage for credentials; choose appropriate local credential storage when packaging is selected.

Sanitize query/analysis provider errors consistently with interpretation errors: [query_service.py:72](../backend/services/query_service.py#L72), [aggregation_service.py:113](../backend/services/aggregation_service.py#L113). User-facing failures should not contain raw provider exception text.

Completion: a dataset marked local-only makes no outbound AI request even with a configured key. A failed AI request leaves deterministic charts available. Configuring and clearing a key works without editing source files.

### 6. A real installation/release path and automated quality gate

**Payoff: high for distribution. Scope: medium initially; packaging larger.**

The current setup is developer-oriented: separate Python/Node environments, backend reload mode and Next development server in [README.md:96](../README.md#L96). There is no checked-in CI/release workflow. Production build/start scripts exist in [frontend/package.json](../frontend/package.json#L5), but no combined local launcher. Frontend builds use Google font fetching at [layout.tsx:2](../frontend/src/app/layout.tsx#L2).

First add a single verification command and CI running locked installs, backend/eval checks, frontend helper/type/lint/build checks, and a small full-app smoke flow. Add supported-platform fresh-install checks and production startup/shutdown instructions. Then provide a local launcher/package, backend readiness checks, port handling, safe updates and accessible diagnostics. Bundle fonts or define a system fallback for offline builds. Account authentication is not a primary next step for this single-user local scope.

Expand regression evidence beyond current synthetic shapes: locale/text-heavy/wide/sparse inputs, high cardinality, mixed units, large decimals/integers, late invalid rows, timezone/resolution, quoted fields, and engine-threshold cases. Add generated/differential cases across both engines and chunk sizes; independently calculate expected analytics rather than only asserting engine agreement. Include user-approved representative exports, with private data handled locally.

Completion: a clean download on each supported OS can install, launch, upload, build a chart, reopen work and export without developer troubleshooting. Releases cannot bypass correctness checks; performance budgets include representative formats, not only the million-row benchmark.

### 7. Finish resource controls, progress and cancellation

**Payoff: high for reliability on ordinary computers. Scope: medium.**

The pandas path materializes a whole frame before checking row/column limits: [csv_ingestion.py:55](../backend/services/csv_ingestion.py#L55). A short-line CSV can contain many rows below the byte threshold, and raw/cleaning copies multiply allocations. Existing two-ingestion and DuckDB limits help but do not bound whole-process RAM or total retained disk use.

Check rows/cells and structural limits during reading for both paths. Add global retained-byte/disk budgets and predictable capacity responses. Use ingestion jobs with observable phases and cooperative cancel/deadline handling, rather than making the browser wait with an upload spinner. Upload has no signal in [api.ts:93](../frontend/src/lib/api.ts#L93); the chat disables submission while running and has no cancel button at [ChatInterface.tsx:68](../frontend/src/components/ChatInterface.tsx#L68). Aborting a browser request alone does not stop backend work. Add shutdown cleanup and safe recovery/cleanup of abandoned temporary imports.

Completion: oversized short-line files, low free disk space, concurrent uploads, query timeout and cancellation have bounded, understandable outcomes. Cancellation releases owned resources without closing another operation's active dataset.

### 8. Bound report export memory and verify actual output

**Payoff: medium now; high if reporting is core. Scope: medium.**

PDF export captures the entire dashboard at 2× dimensions before splitting pages: [exportReport.ts:86](../frontend/src/lib/exportReport.ts#L86). Tall dashboards multiply canvas and encoded-image allocations. Capture/render bounded pages or widgets, establish export size limits, and test axes, wrapping, filters/units, mixed charts, multi-page layouts and cancellation. Historical clipping observations should be rechecked on current code; they are not claimed as newly reproduced defects here.

Completion: representative long dashboards export with measured memory limits and verified page fidelity; failures remain actionable and do not lose the workspace.

### 9. Focused repository cleanup

**Payoff: modest, cheap, worth doing alongside the first changes.**

- Remove the unused mean/mode-imputation function [data_janitor.py:295](../backend/modules/data_janitor.py#L295) and its exports in [modules/__init__.py](../backend/modules/__init__.py#L10). Current ingestion never calls it; it contradicts the null-preservation policy and invites accidental reuse.
- Remove `html2canvas` if import/build checks continue to confirm no caller; export currently uses `html-to-image`. Update the lockfile with the removal.
- Remove confirmed-unused starter SVGs under `frontend/public/` and replace the default [frontend README](../frontend/README.md) with actual project guidance.
- Refresh the main README and storage comments: “all datasets live in memory” is stale after disk ingestion. Link current reports prominently and label older audits as historical.
- Keep test fixtures, holdout datasets and benchmark evidence. Combined benchmark/eval result directories are only roughly half a megabyte. No large committed build output was found. `.next`, `node_modules`, virtual environments and the large taxi CSV are ignored local files; deleting them does not improve the shipped runtime.

## Recommended execution order

Start with import structure/dialect validation, shared parsing semantics and engine-parity tests. Add reviewed schema application next, including precise count contracts and explicit AI sharing. Build durable workspaces and release automation around that stable import/schema contract. Finish job cancellation/global resource budgets and report export validation before broad distribution. Small cleanup can accompany each stage; broad rewrites and mass deletion are not the priority.

A release should demonstrate correct imports across representative formats, reversible confirmed conversions, consistent bounded analytics, reopenable work, intentional data sharing, supported-platform startup, actionable errors, and repeatable checks. Faster uploads and a newer model alone do not establish those properties.

Verification note: parser reproductions used synthetic data. One timestamp-check subagent inadvertently ran the optional summary phase by selecting synchronous enrichment; it may have contacted the configured provider using that synthetic sample. No user dataset or credentials were printed. Other review work was source inspection and offline parser checks; no full new test run was needed because implementation was unchanged.
