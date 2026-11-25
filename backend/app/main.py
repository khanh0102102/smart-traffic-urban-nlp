"""FastAPI application entry point for the ST-NLP backend."""
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes.auth import router as auth_router
from app.api.routes.health import router as health_router
from app.core.config import settings

ROOT_DIR = Path(__file__).resolve().parents[2]
FRONTEND_DIR = ROOT_DIR / "frontend"

app = FastAPI(
    title=settings.app_name,
    version="0.3.0",
    description="Backend API for the Smart Traffic & Urban Incident Classifier.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)

if (FRONTEND_DIR / "assets").is_dir():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIR / "assets"), name="frontend-assets")

app.include_router(health_router, prefix="/api/v1")
app.include_router(auth_router, prefix="/api/v1")


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/citizen/", status_code=307)


@app.get("/citizen/", include_in_schema=False)
def citizen_portal() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "citizen" / "index.html")


@app.get("/admin/", include_in_schema=False)
def admin_portal() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "admin" / "index.html")
