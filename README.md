# Analytico

Analytico is a local-first analytics application that takes users from CSV exploration to exportable reporting. Manual analytics work without an AI key; optional AI features send selected dataset context to the configured provider.

- **Explore Mode:** upload data, ask questions in natural language, or build charts manually.
- **Dashboard Mode:** pin charts, drag/resize widgets on a snap grid, and assemble a report layout.
- **Export Workflow:** download chart assets (PNG/SVG) or export a multi-page PDF dashboard report.

https://github.com/user-attachments/assets/4c3df7ed-f755-4f2b-91f6-d5cb18f45186

## Product Direction

The agreed vision, performance targets, trust contract and incremental roadmap live in [the product goal](docs/product-goal.md). The first milestone is [a dependable upload-to-answer flow](docs/first-milestone.md); its [first real-data scorecard](docs/real-acceptance-2026-10-02.md) includes accepted fixes, rejected prompt experiments and remaining release checks. Targets are acceptance goals, not claims of current performance.

## The Stack

- **Backend:** Python, FastAPI, DuckDB, Pandas
- **Frontend:** TypeScript, Next.js, Tailwind CSS
- **Visualization:** Recharts, Framer Motion, react-grid-layout
- **AI:** optional OpenAI integration (default model: GPT-6 Luna)
- **Export:** html-to-image + jsPDF

---

## Core Product Capabilities

### 1) Data Ingestion

- CSV uploads detect file settings and prepare automatically, with full-file validation before registration. The optional Review data panel contains file settings, column edits, AI proposals, samples and cleaning details. Failed imports remain available for settings correction without automatically opening a panel.
- Large-file DuckDB ingestion uses up to four threads per dataset, capped to available CPUs. Set `ANALYTICO_DUCKDB_THREADS=1` or `2` in `backend/.env` for a smaller resource budget (accepted range: 1–8). More threads increase peak memory; the 64 MB DuckDB execution budget is not a total process-memory limit.
- Included Gapminder demo; optional taxi demo when its CSV is installed.
- Stable source column keys and conservative format detection for numeric, currency, percentage, and date fields. With AI column semantic analysis enabled, compatible roles and readable display labels apply in the background; charts are available as soon as local preparation finishes.
- Missing observations remain null; ingestion retains the original CSV bytes separately from the parsed view for the current session.
- Review column parsing, roles, units, formats, aggregation, and display labels. Parsing/schema edits rebuild from source and invalidate previous charts; label-only edits update metadata without reingesting the CSV.

### 2) AI + Manual Charting

- Chat-to-chart flow with validated chart configs and filters. Overall totals, ordinary averages, medians, minima, maxima, and counts appear as a single bar with a value label; grouped questions retain their comparison charts.
- Overall row counts count matching rows; named-column counts count non-missing values. Filters apply before aggregation. Ambiguous business meanings, distinct counts, weighted rates, and unsupported transformations return clarification.
- Both AI and manual charts use the same deterministic aggregation path. Generated Python execution is not supported.
- Manual chart builder with aggregation support:
  - `sum`, `mean`, `median`, `count`, `min`, `max`
- On-demand AI chart analysis for current view.

### 3) Filtering + Drilldown

- Structured filter operators:
  - `eq`, `gt`, `lt`, `gte`, `lte`, `contains`
- Drilldown uses full structured filter state, including AI-generated filters.
- Graceful handling of invalid AI chart configs.

### 4) Data Quality + Cleaning Transparency

- Dataset quality score with clear completeness formula.
- Missing-value breakdown and profile summaries.
- Cleaning Report view showing what was changed during ingestion.

### 5) Dashboard Command Center

- Pin charts directly from Explore.
- Drag, resize, and snap widgets on a responsive grid.
- Per-dataset dashboard persistence in local storage.
- Collapsible data summary panel to maximize dashboard space.

### 6) Export

- Chart-level exports: PNG, SVG.
- Dashboard-level export: multi-page A4 landscape PDF.
- Export capture scoped to dashboard surface to avoid UI overlay artifacts.

---

## Typical Workflow

1. Optionally enable AI column semantic analysis, then upload a CSV or load the included Gapminder demo.
2. Local preparation runs automatically; charts become available first, then compatible AI role and display-label updates apply in the background.
3. Ask a question in chat or build a chart manually. Open Review data whenever you want to inspect or edit parsing and column choices.
4. Refine with filters and drilldown.
5. Pin charts to Dashboard and arrange layout.
6. Export the dashboard as a PDF report.

---

## Project Structure

```text
analytico/
├── backend/
│   ├── main.py
│   ├── modules/
│   ├── models.py
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── app/
│   │   ├── components/
│   │   ├── context/
│   │   ├── lib/
│   │   └── types/
│   └── package.json
└── README.md
```

---

## Running Locally

### Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install --require-hashes --only-binary=:all: -r requirements.txt
cp .env.example .env  # First setup only; add your own API key if using AI.
uvicorn main:app --host 127.0.0.1 --reload
```

The backend lock targets Python 3.11 or newer; the tested environment is Python 3.14 on macOS. Runtime and development requirements pin all resolved packages and verify download hashes. See [dependency maintenance](docs/dependency-locking.md) for controlled upgrades and platform limitations.

### Frontend

```bash
cd frontend
npm ci
npm run dev
```

Open [http://localhost:3000](http://localhost:3000)

Set `OPENAI_API_KEY` in the ignored `backend/.env` for chart planning and descriptions. Use the upload-time **AI column semantic analysis** toggle for GPT-6 Luna schema roles and readable labels. It is off by default; `COLUMN_INTERPRETER=luna` configures a default for callers that omit the per-upload choice. The batched schema request is pinned to `gpt-6-luna`; chart planning and descriptions default to the same model through `OPENAI_MODEL`. The older Jev provider remains an optional configuration, but the upload toggle selects Luna. See [column interpretation](docs/column-interpretation.md) for sampling limits and safety checks. With an OpenAI key configured, ingestion can still send summary context automatically; an explicit per-dataset privacy switch is pending. Upload, profiling, filtering, and manual charts remain available without it. Unsupported advanced questions return a clarification rather than running generated code.

The backend is intended to run on localhost. It has no account/authentication system; do not expose it as a public server. The dataset registry is in memory and expires after inactivity. Small datasets use pandas in memory; CSVs of at least 8 MiB use temporary disk-backed DuckDB storage by default. Neither path provides a durable workspace: datasets cannot be reopened through the app after a backend restart. History and dashboard snapshots are stored in the browser, scoped to the dataset. Retaining raw data is not yet a durable workspace or backup mechanism.

Gapminder is included in the repository. The taxi demo requires the separate CSV named in `backend/core/config.py`; it is not included in a fresh download.

## Validation

Backend checks are deterministic and mock AI calls:

```bash
cd backend
pip install --require-hashes --only-binary=:all: -r requirements-dev.txt
python -m unittest discover -s tests -v
```

Frontend checks:

```bash
cd frontend
npm run lint
npx tsc --noEmit
npm run test:helpers
npm run build
```

## Before changing AI interpretation

The synthetic interpretation benchmark defines expected column roles, units, parsing policies, aggregation recommendations, and clarification decisions. It includes ambiguous cases where guessing would be wrong. Its richer interpretation vocabulary does not add new runtime operations such as weighted rates or snapshot aggregation.

```bash
python3 backend/evals/evaluate_interpretation.py --self-check
python3 backend/evals/evaluate_interpretation.py --predictions backend/evals/baseline_current_repo.json
```

See [benchmark instructions](docs/interpretation-benchmark.md) for scoring future Jev/Luna predictions and regenerating the offline baseline. See [dependency audit](docs/dependency-audit-2026-10-01.md) for package changes and residual findings. The [original repository audit](docs/production-readiness-audit-2026-10-01.md) is a historical pre-fix document; [completed fixes](docs/local-safety-fixes-2026-10-01.md) records the verified implementation.

The latest [Luna improvement results](docs/luna-improvement-results-2026-10-01.md) include independent holdout runs and deterministic safety regressions.

The [performance follow-up](docs/performance-improvements-2026-10-02.md) documents shared statistics, bounded chart/interpretation caches, measurement scope, and reproducible offline benchmarks. Cache limits do not replace large-file resource budgets or durable storage.

Large CSVs now load directly into DuckDB as source text before full-column conversion and profiling. Complete-file validation remains enabled. Unsupported encodings, single-column files and native-reader errors use the compatible pandas chunk reader; small files keep their existing pandas path. See the [native loader comparison](docs/ingestion-patterns-benchmark-2026-10-02.md) for measured performance and scope.

The [shared column policy and rule audit](docs/deterministic-policy-benchmark-2026-10-03.md) records the current deterministic before/after, full real-data checks, and accuracy/coverage tradeoffs.

Optional **AI column semantic analysis** uses one bounded GPT-6 Luna request per supported schema. Local charts are ready first; compatible role updates apply automatically in the background. Review data remains optional. Units, parsing, aggregation and original values are preserved. See [benchmark artifact policy](docs/benchmark-artifacts.md) for retained evidence and cleanup.

## Latest verified improvements

- **Background schema analysis and readable labels:** one bounded GPT-6 Luna request suggests roles and display names while preserving source keys and values. Strong, compatible suggestions apply automatically; Review data remains optional. See the [fresh multi-dataset accuracy baseline](docs/fresh-accuracy-2026-10-03.md).
- **Overall-answer charts:** total/average/count questions use the existing chart, filter, analysis, and drill-down flow. Full-file calculations passed 12 independent checks across tips, diamonds, and taxi; nine live Luna planner checks also passed. See [overall-chart validation](docs/overall-charts-2026-10-03.md).
- **Measured ingestion parallelism:** on the benchmark machine, four DuckDB threads reduced median taxi preparation from 13.03s to 10.84s and retail preparation from 3.36s to 2.82s, with increased peak memory. These are three-run local-engine measurements, excluding upload transfer, AI, and browser rendering. Full-source fingerprints matched in all 144 experimental imports. Only the CPU-capped four-thread setting was promoted; experimental scan pruning, concurrent batches, multiprocessing, and early-sample AI remain unshipped. See [timings, memory, and methodology](docs/ingestion-parallelism-2026-10-03.md).
- **Release checks:** 245 deterministic backend tests, frontend TypeScript checks, and helper/rendering checks passed on 2026-10-06. This does not establish universal accuracy or replace packaging, restart recovery, and cross-platform release testing.
