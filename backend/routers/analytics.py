"""Analytics and query API routes."""

from fastapi import APIRouter, HTTPException

from models import AggregateRequest, ChartResponse, DrillDownRequest, QueryRequest
from services.aggregation_service import records_to_json, run_aggregate
from services.query_service import run_query
from storage import get_dataset
from utils.filtering import FilterValidationError, apply_filters

router = APIRouter()


@router.post("/aggregate", response_model=ChartResponse)
def aggregate_endpoint(request: AggregateRequest):
    return run_aggregate(request)


@router.post("/query", response_model=ChartResponse)
def query_endpoint(request: QueryRequest):
    return run_query(request)


@router.post("/drilldown")
def drilldown_endpoint(request: DrillDownRequest):
    ds = get_dataset(request.dataset_id)
    try:
        filtered, _ = apply_filters(ds.df, request.filters)
    except FilterValidationError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    rows = records_to_json(filtered.head(request.limit))
    return {
        "data": rows,
        "total_rows": len(filtered),
        "limit": request.limit,
    }
