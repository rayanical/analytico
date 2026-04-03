"""Health and root endpoints."""

from fastapi import APIRouter

router = APIRouter()


@router.get("/")
async def root():
    return {"status": "ok", "version": "5.0.0", "architecture": "modular"}
