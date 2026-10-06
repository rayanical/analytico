"""Ingestion-related API routes."""

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from pydantic import ValidationError
import pandas as pd

from core.config import DEMO_DATASETS
from models import UploadResponse, ImportPreviewRequest, ImportConfirmRequest, SchemaApplyRequest
from modules.import_policy import ImportSettings, CSVStructureError
from services.csv_ingestion import ingest_csv
from services.enrichment_service import get_enrichment_status
from storage import get_dataset
from utils.errors import friendly_ingestion_error_message
from services.import_preview import stage_import, preview_import, confirm_import, cancel_import
from services.schema_review import get_schema, apply_schema

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


def _ingest(source, filename: str, endpoint: str, ai_column_analysis: bool = False):
    try:
        return ingest_csv(source, filename, endpoint, ai_column_analysis=ai_column_analysis)
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="Empty CSV.")
    except HTTPException:
        raise
    except CSVStructureError as error:
        raise HTTPException(413 if error.limit_exceeded else 400, str(error)) from error
    except Exception as error:
        status, detail = friendly_ingestion_error_message(error)
        raise HTTPException(status_code=status, detail=detail) from error


@router.post("/upload", response_model=UploadResponse)
def upload_csv(file: UploadFile = File(...), ai_column_analysis: bool = Form(default=False)):
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a CSV file.")
    return _ingest(file.file, file.filename, "/upload", ai_column_analysis)


@router.post("/load-demo", response_model=UploadResponse)
def load_demo_dataset(dataset: str = Query(default="taxi"), ai_column_analysis: bool = Query(default=False)):
    selected_demo = DEMO_DATASETS.get(dataset)
    if not selected_demo:
        available = ", ".join(sorted(DEMO_DATASETS.keys()))
        raise HTTPException(status_code=400, detail=f"Unknown demo dataset '{dataset}'. Available: {available}")
    filename, demo_path = selected_demo["filename"], selected_demo["path"]
    if not demo_path.exists():
        raise HTTPException(status_code=500, detail=f"Demo dataset file not found: {filename}")
    return _ingest(demo_path, filename, "/load-demo", ai_column_analysis)


@router.post("/imports/preview")
def stage_csv_preview(file: UploadFile = File(...), settings: str | None = Form(default=None)):
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "Please choose a CSV file.")
    try:
        selected = ImportSettings.model_validate_json(settings) if settings else None
    except ValidationError as error:
        raise HTTPException(422, "Invalid import settings.") from error
    return stage_import(file.file, file.filename, selected)


@router.post("/imports/demo")
def stage_demo_preview(dataset: str = Query(default="gapminder")):
    demo = DEMO_DATASETS.get(dataset)
    if demo is None:
        raise HTTPException(400, "Unknown demo dataset.")
    if not demo["path"].exists():
        raise HTTPException(404, "This demo CSV is not installed locally.")
    return stage_import(demo["path"], demo["filename"])


@router.post("/imports/{import_id}/preview")
def refresh_csv_preview(import_id: str, request: ImportPreviewRequest):
    return preview_import(import_id, request.settings)


@router.post("/imports/{import_id}/confirm", response_model=UploadResponse)
def confirm_csv_preview(import_id: str, request: ImportConfirmRequest):
    return confirm_import(import_id, request.settings, request.column_overrides,
                          ai_column_analysis=request.ai_column_analysis)


@router.delete("/imports/{import_id}")
def cancel_csv_preview(import_id: str):
    return cancel_import(import_id)


@router.get("/datasets/{dataset_id}/schema")
def review_dataset_schema(dataset_id: str):
    return get_schema(dataset_id)


@router.post("/datasets/{dataset_id}/schema", response_model=UploadResponse)
def apply_dataset_schema(dataset_id: str, request: SchemaApplyRequest):
    return apply_schema(dataset_id, request.expected_version, request.column_overrides, request.settings)
