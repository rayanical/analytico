"""
Analytico Backend - FastAPI Server
Self-service data visualization with AI-powered chart generation
"""

import json
import os
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Analytico API",
    description="AI-powered data visualization backend",
    version="1.0.0"
)

# Configure CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Constants
MAX_ROWS = 1000  # Limit rows to avoid localStorage issues on frontend


# Pydantic Models
class QueryRequest(BaseModel):
    user_prompt: str
    columns: list[str]
    data_summary: dict[str, Any]


class ChartConfig(BaseModel):
    xAxisKey: str
    yAxisKeys: list[str]
    chartType: str
    title: str


class UploadResponse(BaseModel):
    data: list[dict[str, Any]]
    columns: list[str]
    summary: dict[str, Any]
    row_count: int
    truncated: bool


# System prompt for OpenAI
SYSTEM_PROMPT = """You are a data visualization assistant. You will receive a user question, a list of dataset columns, and a data summary. You must return a raw JSON object (no markdown) that maps the user's request to the correct columns for a Recharts graph.

Determine which column is the categorical key (xAxisKey) and which are the numerical values (yAxisKeys).

Rules:
1. xAxisKey MUST be a column that exists in the provided columns list
2. yAxisKeys MUST only contain columns that exist in the provided columns list
3. yAxisKeys should only contain numeric columns (check the data_summary for numeric stats)
4. chartType must be one of: "bar", "line", "area", "composed"
5. Generate a descriptive title based on what the chart will show

Example Output:
{"xAxisKey": "month", "yAxisKeys": ["revenue", "cost"], "chartType": "bar", "title": "Revenue vs Cost over Time"}

Return ONLY the JSON object, no markdown formatting, no code blocks, no explanation."""


@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "ok", "message": "Analytico API is running"}


@app.post("/upload", response_model=UploadResponse)
async def upload_csv(file: UploadFile = File(...)):
    """
    Upload a CSV file and get parsed data with statistics.
    
    - Accepts .csv files only
    - Parses with Pandas
    - Replaces NaN with None
    - Returns first MAX_ROWS rows to avoid localStorage limits
    - Includes summary statistics for numeric columns
    """
    # Validate file type
    if not file.filename or not file.filename.endswith('.csv'):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Please upload a CSV file."
        )
    
    try:
        # Read CSV with Pandas
        df = pd.read_csv(file.file)
        
        # Check if DataFrame is empty
        if df.empty:
            raise HTTPException(
                status_code=400,
                detail="The uploaded CSV file is empty."
            )
        
        # Track if we truncated the data
        original_count = len(df)
        truncated = original_count > MAX_ROWS
        
        # Limit rows to avoid localStorage issues
        if truncated:
            df = df.head(MAX_ROWS)
        
        # Replace NaN with None for JSON compatibility
        df = df.fillna(0)  # Replace NaN with 0 for numeric columns
        df = df.where(pd.notnull(df), None)  # Replace remaining with None
        
        # Get column names
        columns = df.columns.tolist()
        
        # Generate summary statistics for numeric columns
        summary = {}
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        for col in numeric_cols:
            summary[col] = {
                "min": float(df[col].min()) if pd.notnull(df[col].min()) else 0,
                "max": float(df[col].max()) if pd.notnull(df[col].max()) else 0,
                "mean": float(df[col].mean()) if pd.notnull(df[col].mean()) else 0,
                "type": "numeric"
            }
        
        # Add non-numeric columns to summary with type info
        for col in columns:
            if col not in numeric_cols:
                unique_count = df[col].nunique()
                summary[col] = {
                    "type": "categorical",
                    "unique_values": unique_count,
                    "sample_values": df[col].head(5).tolist()
                }
        
        # Convert to list of dictionaries
        data = df.to_dict(orient='records')
        
        return UploadResponse(
            data=data,
            columns=columns,
            summary=summary,
            row_count=len(data),
            truncated=truncated
        )
        
    except pd.errors.EmptyDataError:
        raise HTTPException(
            status_code=400,
            detail="The uploaded CSV file is empty or malformed."
        )
    except pd.errors.ParserError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to parse CSV: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while processing the file: {str(e)}"
        )


@app.post("/query", response_model=ChartConfig)
async def query_data(request: QueryRequest):
    """
    Generate a chart configuration from a natural language query.
    
    - Accepts user prompt, column names, and data summary
    - Uses OpenAI GPT-4o-mini to determine chart configuration
    - Returns Recharts-compatible configuration
    """
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(
            status_code=500,
            detail="OpenAI API key not configured. Please set OPENAI_API_KEY environment variable."
        )
    
    try:
        # Prepare context for OpenAI
        user_message = f"""User Question: {request.user_prompt}

Available Columns: {json.dumps(request.columns)}

Data Summary (with stats for numeric columns):
{json.dumps(request.data_summary, indent=2)}

Based on the user's question and the available data, generate a chart configuration."""

        # Call OpenAI API
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ],
            temperature=0.3,  # Lower temperature for more consistent output
            max_tokens=500
        )
        
        # Extract response content
        content = response.choices[0].message.content
        if not content:
            raise HTTPException(
                status_code=500,
                detail="OpenAI returned an empty response."
            )
        
        # Parse JSON response
        try:
            # Clean up response if it has markdown code blocks
            cleaned_content = content.strip()
            if cleaned_content.startswith("```"):
                # Remove markdown code blocks
                lines = cleaned_content.split("\n")
                cleaned_content = "\n".join(lines[1:-1])
            
            chart_config = json.loads(cleaned_content)
        except json.JSONDecodeError as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to parse AI response as JSON: {str(e)}. Response was: {content}"
            )
        
        # Validate required fields
        required_fields = ["xAxisKey", "yAxisKeys", "chartType", "title"]
        for field in required_fields:
            if field not in chart_config:
                raise HTTPException(
                    status_code=500,
                    detail=f"AI response missing required field: {field}"
                )
        
        # Validate chart type
        valid_types = ["bar", "line", "area", "composed"]
        if chart_config["chartType"] not in valid_types:
            chart_config["chartType"] = "bar"  # Default to bar if invalid
        
        # Validate columns exist
        if chart_config["xAxisKey"] not in request.columns:
            raise HTTPException(
                status_code=500,
                detail=f"AI selected invalid xAxisKey: {chart_config['xAxisKey']}"
            )
        
        invalid_y_keys = [k for k in chart_config["yAxisKeys"] if k not in request.columns]
        if invalid_y_keys:
            raise HTTPException(
                status_code=500,
                detail=f"AI selected invalid yAxisKeys: {invalid_y_keys}"
            )
        
        return ChartConfig(**chart_config)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error calling OpenAI API: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
