"""FastAPI application entry point for the Phase 1 skeleton."""

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes.health import router as health_router

ROOT_DIR = Path(__file__).resolve().parents[2]
FRONTEND_DIR = ROOT_DIR / "frontend"

app = FastAPI(
    title="ST-NLP API",
    version="0.1.0",
    description="Starter API for the Smart Traffic & Urban Incident Classifier.",
)

if (FRONTEND_DIR / "assets").is_dir():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIR / "assets"), name="frontend-assets")

app.include_router(health_router, prefix="/api/v1")


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    """Send the site root to the citizen portal."""
    return RedirectResponse(url="/citizen/", status_code=307)


@app.get("/citizen/", include_in_schema=False)
def citizen_portal() -> FileResponse:
    """Serve the citizen-facing starter page."""
    return FileResponse(FRONTEND_DIR / "citizen" / "index.html")


@app.get("/admin/", include_in_schema=False)
def admin_portal() -> FileResponse:
    """Serve the admin-facing starter page on a separate URL."""
    return FileResponse(FRONTEND_DIR / "admin" / "index.html")
