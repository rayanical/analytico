# Analytico

Analytico is a full-stack AI analytics application that takes users from quick exploration to exportable reporting.

- **Explore Mode:** upload data, ask questions in natural language, or build charts manually.
- **Dashboard Mode:** pin charts, drag/resize widgets on a snap grid, and assemble a report layout.
- **Export Workflow:** download chart assets (PNG/SVG) or export a multi-page PDF dashboard report.



https://github.com/user-attachments/assets/4c3df7ed-f755-4f2b-91f6-d5cb18f45186



## The Stack

- **Backend:** Python, FastAPI, Pandas
- **Frontend:** TypeScript, Next.js, Tailwind CSS
- **Visualization:** Recharts, Framer Motion, react-grid-layout
- **AI:** OpenAI GPT-4o-mini
- **Export:** html-to-image + jsPDF

---

## Core Product Capabilities

### 1) Data Ingestion

- CSV upload with automatic cleaning and profiling.
- Instant 1M-row demo dataset loading (no browser upload wait).
- Deterministic format detection for numeric, currency, percentage, and date fields.

### 2) AI + Manual Charting

- Chat-to-chart flow with validated chart configs.
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

1. Upload a CSV or load the 1M-row demo dataset.
2. Ask a question in chat or build a chart manually.
3. Refine with filters and drilldown.
4. Pin charts to Dashboard and arrange layout.
5. Export the dashboard as a PDF report.

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
pip install -r requirements.txt
uvicorn main:app --reload
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000)

