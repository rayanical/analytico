"""Analytics and query API routes."""

from fastapi import APIRouter
import pandas as pd

from models import AggregateRequest, ChartResponse, DrillDownRequest, QueryRequest
from services.aggregation_service import run_aggregate
from services.query_service import run_query
from storage import get_dataset
from utils.filtering import apply_filters

router = APIRouter()


@router.post("/aggregate", response_model=ChartResponse)
async def aggregate_endpoint(request: AggregateRequest):
    return run_aggregate(request)


@router.post("/query", response_model=ChartResponse)
async def query_endpoint(request: QueryRequest):
    return run_query(request)


@router.post("/drilldown")
async def drilldown_endpoint(request: DrillDownRequest):
    ds = get_dataset(request.dataset_id)
    df = ds.df

    filtered, _ = apply_filters(df, request.filters)

    data = filtered.head(request.limit).to_dict(orient="records")
    for row in data:
        for k in row:
            if pd.isna(row[k]):
                row[k] = None

    return {
        "data": data,
        "total_rows": len(filtered),
        "limit": request.limit,
    }
