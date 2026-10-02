"""Ingestion-related API routes."""

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
import pandas as pd

from core.config import DEMO_DATASETS
from models import UploadResponse
from services.csv_ingestion import ingest_csv
from services.enrichment_service import get_enrichment_status
from storage import get_dataset
from utils.errors import friendly_ingestion_error_message

router = APIRouter()


@router.get("/validate/{dataset_id}")
def validate_dataset(dataset_id: str):
    """Check whether a session dataset is still available."""
    try:
        ds = get_dataset(dataset_id)
        return {"valid": True, "filename": ds.filename}
    except HTTPException as error:
        if error.status_code == 404:
            return {"valid": False}
        raise


@router.get("/enrichment/{dataset_id}")
def enrichment_status(dataset_id: str):
    dataset = get_dataset(dataset_id)
    return get_enrichment_status(dataset_id, dataset.cache_version)


def _ingest(source, filename: str, endpoint: str):
    try:
        return ingest_csv(source, filename, endpoint)
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="Empty CSV.")
    except HTTPException:
        raise
    except Exception as error:
        status, detail = friendly_ingestion_error_message(error)
        raise HTTPException(status_code=status, detail=detail) from error


@router.post("/upload", response_model=UploadResponse)
def upload_csv(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a CSV file.")
    return _ingest(file.file, file.filename, "/upload")


@router.post("/load-demo", response_model=UploadResponse)
def load_demo_dataset(dataset: str = Query(default="taxi")):
    selected_demo = DEMO_DATASETS.get(dataset)
    if not selected_demo:
        available = ", ".join(sorted(DEMO_DATASETS.keys()))
        raise HTTPException(status_code=400, detail=f"Unknown demo dataset '{dataset}'. Available: {available}")
    filename, demo_path = selected_demo["filename"], selected_demo["path"]
    if not demo_path.exists():
        raise HTTPException(status_code=500, detail=f"Demo dataset file not found: {filename}")
    return _ingest(demo_path, filename, "/load-demo")
