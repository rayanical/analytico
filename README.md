# Analytico

A full-stack AI charting platform designed to make data visualization zero-friction. Upload a CSV, ask a question in plain English, and get an instant chart. No manual configuration required, unless wanted.

### The Stack
- **Engine:** Python / FastAPI / Pandas
- **Interface:** TypeScript / Next.js / Tailwind CSS
- **Visualization:** Recharts / Framer Motion
- **Intelligence:** OpenAI GPT-4o-mini

---

## Technical Features

### 1. Smart Data Ingestion ("Data Janitor")

The backend includes a Python module that automatically cleans uploaded CSVs without user intervention:

- **Header Normalization**: Converts verbose survey-style headers into clean column names
  - `"What is your annual salary?"` → `annual_salary`
  - `"How old are you?"` → `age`
  
- **Type Detection**: Identifies currency (`$45,000` → numeric), percentages, and dates by analyzing column content patterns

- **Missing Value Handling**: Automatically fills missing numeric values with zeros and logs the action for transparency

```python
# The 'Data Janitor' uses multi-phase regex cleanup
def normalize_header(header: str) -> str:
    # 1. Full replacements for known survey patterns
    # 2. Prefix removals (e.g., 'what_is_your_')
    # 3. Suffix removals (e.g., '_choose_all_that_apply')
    # 4. 20-character truncation for clean UI display
    ...
```

### 2. Semantic Column Analysis

Before generating visualizations, the system classifies each column into semantic types to prevent nonsensical aggregations:

| Type | Example | Behavior |
|------|---------|----------|
| **Metric** | Revenue, Salary | Can be summed, averaged |
| **Categorical** | Department, Status | Used for grouping, auto-switches to COUNT |
| **Identifier** | Employee ID, Zip Code | Blocked from sum/mean operations |
| **Temporal** | Date, Month | Used as X-axis, auto-resampled |

This prevents common mistakes like summing zip codes or averaging ID numbers.

### 3. Natural Language Querying

Users describe what they want in plain English. The system:

1. Sends the user's question + dataset metadata to GPT-4o-mini
2. Receives a structured JSON response with chart configuration
3. Validates the response against actual column names
4. Renders the chart dynamically

```
User: "Show me average salary by industry"

→ AI returns:
{
  "xAxisKey": "industry",
  "yAxisKeys": ["annual_salary"],
  "aggregation": "mean",
  "chartType": "bar",
  "title": "Average Salary by Industry"
}
```

### 4. Smart Aggregation

The backend handles edge cases that would otherwise produce unusable charts:

- **High Cardinality**: Groups categories beyond the top 19 into "Others"
- **Date Resampling**: Automatically groups timestamps into years/months/weeks based on date range
- **Aggregation Passthrough**: Respects the requested aggregation type (mean, sum, count) through all processing steps

### 5. Dynamic Visualization

The frontend renders charts using Recharts with:

- Responsive containers that adapt to screen size
- Memoized tooltip and legend components for performance
- Automatic number formatting (currency, percentages, compact notation)
- Framer Motion animations for smooth transitions

---

## Project Structure

```
analytico/
├── backend/
│   ├── main.py          # FastAPI app, all endpoints
│   ├── requirements.txt
│   └── venv/
├── frontend/
│   ├── src/
│   │   ├── app/         # Next.js app router
│   │   ├── components/  # React components
│   │   ├── context/     # Global state (DataContext)
│   │   └── lib/         # API client
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

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/upload` | Upload CSV, returns cleaned data profile |
| `POST` | `/query` | Natural language → chart configuration |
| `POST` | `/aggregate` | Manual chart configuration |
| `GET` | `/` | Health check |


---

## Future Improvements

- [ ] Support for Excel files
- [ ] Chart export (PNG, SVG)
- [ ] Saved queries / dashboard persistence
- [ ] User authentication
- [ ] Improved recommendations
- [ ] Better dataset cleaning

