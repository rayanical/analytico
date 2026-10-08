"""
Analytico Backend V5 - Thin FastAPI entrypoint.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from core.config import APP_DESCRIPTION, APP_TITLE, APP_VERSION, CORS_ALLOW_ORIGINS
from routers.analytics import router as analytics_router
from routers.health import router as health_router
from routers.ingestion import router as ingestion_router
from services.csv_ingestion import preload_ingestion_engine


@asynccontextmanager
async def lifespan(_app):
    preload_ingestion_engine()
    yield

app = FastAPI(
    title=APP_TITLE,
    description=APP_DESCRIPTION,
    version=APP_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(ingestion_router)
app.include_router(analytics_router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
