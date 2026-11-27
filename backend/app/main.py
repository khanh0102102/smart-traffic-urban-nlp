"""FastAPI application entry point for the ST-NLP backend."""
from html import escape
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes.auth import router as auth_router
from app.api.routes.health import router as health_router
from app.api.routes.reports import router as reports_router
from app.api.routes.predictions import router as predictions_router
from app.api.routes.reviews import router as reviews_router
from app.api.routes.dashboard import router as dashboard_router
from app.api.routes.ml_management import router as ml_management_router
from app.core.config import settings

ROOT_DIR = Path(__file__).resolve().parents[2]
FRONTEND_DIR = ROOT_DIR / "frontend"
GOOGLE_CLIENT_ID_PLACEHOLDER = "YOUR_GOOGLE_WEB_CLIENT_ID.apps.googleusercontent.com"

app = FastAPI(
    title=settings.app_name,
    version="0.8.0",
    description="Backend API for the Smart Traffic & Urban Incident Classifier.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)

if (FRONTEND_DIR / "assets").is_dir():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIR / "assets"), name="frontend-assets")

app.include_router(health_router, prefix="/api/v1")
app.include_router(auth_router, prefix="/api/v1")
app.include_router(reports_router, prefix="/api/v1")
app.include_router(predictions_router, prefix="/api/v1")
app.include_router(reviews_router, prefix="/api/v1")
app.include_router(dashboard_router, prefix="/api/v1")
app.include_router(ml_management_router, prefix="/api/v1")


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/citizen/", status_code=307)


@app.get("/citizen/", include_in_schema=False, response_class=HTMLResponse)
def citizen_portal() -> HTMLResponse:
    """Serve the citizen portal with its Google OAuth client ID injected from the environment."""
    html_path = FRONTEND_DIR / "citizen" / "index.html"
    html = html_path.read_text(encoding="utf-8")
    html = html.replace(
        GOOGLE_CLIENT_ID_PLACEHOLDER,
        escape(settings.google_client_id or "", quote=True),
    )
    return HTMLResponse(content=html)


@app.get("/admin/", include_in_schema=False)
def admin_portal() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "admin" / "index.html")
