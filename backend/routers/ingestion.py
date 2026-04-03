"""Ingestion-related API routes."""

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
import pandas as pd

from core.config import DEMO_DATASETS
from models import UploadResponse
from services.ingestion_service import ingest_dataframe
from storage import cleanup_expired
from utils.dataframe_utils import read_csv_fast
from utils.errors import friendly_ingestion_error_message

router = APIRouter()


@router.get("/validate/{dataset_id}")
async def validate_dataset(dataset_id: str):
    """Check if a dataset ID is still valid (exists in memory)."""
    cleanup_expired()
    from storage import DATASETS

    if dataset_id in DATASETS:
        ds = DATASETS[dataset_id]
        ds.touch()
        return {"valid": True, "filename": ds.filename}
    return {"valid": False}


@router.post("/upload", response_model=UploadResponse)
async def upload_csv(file: UploadFile = File(...)):
    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a CSV file.")

    try:
        df = read_csv_fast(file.file)
        if df.empty:
            raise HTTPException(status_code=400, detail="CSV is empty.")
        return ingest_dataframe(df, file.filename, endpoint_name="/upload")
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="Empty CSV.")
    except Exception as e:
        status, detail = friendly_ingestion_error_message(e)
        raise HTTPException(status_code=status, detail=detail)


@router.post("/load-demo", response_model=UploadResponse)
async def load_demo_dataset(dataset: str = Query(default="taxi")):
    selected_demo = DEMO_DATASETS.get(dataset)
    if not selected_demo:
        available = ", ".join(sorted(DEMO_DATASETS.keys()))
        raise HTTPException(status_code=400, detail=f"Unknown demo dataset '{dataset}'. Available: {available}")

    filename = selected_demo["filename"]
    demo_path = selected_demo["path"]
    if not demo_path.exists():
        raise HTTPException(status_code=500, detail=f"Demo dataset file not found: {filename}")

    try:
        df = read_csv_fast(demo_path)
        if df.empty:
            raise HTTPException(status_code=400, detail="CSV is empty.")
        return ingest_dataframe(df, filename, endpoint_name="/load-demo")
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="Empty CSV.")
    except Exception as e:
        status, detail = friendly_ingestion_error_message(e)
        raise HTTPException(status_code=status, detail=detail)
