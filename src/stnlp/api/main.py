"""ST-NLP FastAPI application."""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from stnlp.config import settings
from stnlp.database.init_db import initialize_database
from stnlp.inference.predictor import get_predictor
from stnlp.api.routes import health, predict, reports, admin


@asynccontextmanager
async def lifespan(app: FastAPI):
    initialize_database()
    # Load the selected artifact once at startup. Missing artifacts are reported by readiness
    # and prediction endpoints; the API itself remains available for health checks.
    get_predictor()
    yield


app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    description="API phân loại phản ánh giao thông / đô thị tiếng Việt.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=False,
    allow_methods=["GET", "POST", "PATCH"],
    allow_headers=["Content-Type", "X-Admin-Key"],
)

prefix = settings.api_v1_prefix.rstrip("/")
app.include_router(health.router, prefix=prefix)
app.include_router(predict.router, prefix=prefix)
app.include_router(reports.router, prefix=prefix)
app.include_router(admin.router, prefix=prefix)

ROOT = Path(__file__).resolve().parents[3]
FRONTEND_DIR = ROOT / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(FRONTEND_DIR / "index.html")
