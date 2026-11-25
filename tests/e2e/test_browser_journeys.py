"""Real-browser E2E journey: citizen submission followed by staff review.

The browser test is intentionally opt-in locally and is enabled in the dedicated
GitHub Actions browser-e2e job with RUN_BROWSER_E2E=1.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen
from uuid import uuid4

import pytest

if os.getenv("RUN_BROWSER_E2E") != "1":
    pytest.skip("Browser E2E is run in the dedicated CI job.", allow_module_level=True)

from playwright.sync_api import expect, sync_playwright  # noqa: E402
from sqlalchemy import create_engine, select  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402

from app.core.config import PROJECT_ROOT, settings  # noqa: E402
from app.core.security import create_access_token, hash_password  # noqa: E402
from app.database import models  # noqa: E402,F401
from app.database.base import Base  # noqa: E402
from app.database.models import IncidentCategory, ModelVersion, Prediction, Report, User  # noqa: E402
from app.database.seed import CATEGORIES  # noqa: E402


# Keep the browser flow independent from the CDN. Node tests separately exercise
# the Axios wrapper; this small transport adapter preserves real HTTP/API behavior.
AXIOS_TEST_ADAPTER = r"""
(() => {
  const create = defaults => {
    let requestInterceptor = config => config;
    let responseSuccess = response => response.data;
    let responseFailure = error => Promise.reject(error);
    const client = {
      interceptors: {
        request: { use(handler) { requestInterceptor = handler; } },
        response: {
          use(success, failure) {
            responseSuccess = success;
            responseFailure = failure;
          }
        }
      },
      async request(input) {
        let config = {
          ...defaults,
          ...input,
          headers: { ...(defaults.headers || {}), ...(input.headers || {}) }
        };
        try {
          config = await requestInterceptor(config);
          const url = String(config.baseURL || "") + String(config.url || "");
          const headers = new Headers(config.headers || {});
          let body = config.data;
          if (body !== undefined && !(body instanceof FormData)) {
            if (typeof body !== "string") body = JSON.stringify(body);
            if (!headers.has("Content-Type")) headers.set("Content-Type", "application/json");
          }
          const response = await fetch(url, {
            method: String(config.method || "GET").toUpperCase(),
            headers,
            body
          });
          const data = config.responseType === "blob"
            ? await response.blob()
            : await response.json().catch(() => null);
          if (!response.ok) {
            const error = new Error(data?.detail || ("HTTP " + response.status));
            error.config = config;
            error.response = { status: response.status, data };
            throw error;
          }
          return await responseSuccess({ data, status: response.status, config, headers: {} });
        } catch (error) {
          return responseFailure(error);
        }
      },
      get(url, config = {}) { return this.request({ ...config, method: "GET", url }); },
      post(url, data, config = {}) { return this.request({ ...config, method: "POST", url, data }); },
      patch(url, data, config = {}) { return this.request({ ...config, method: "PATCH", url, data }); },
      delete(url, config = {}) { return this.request({ ...config, method: "DELETE", url }); }
    };
    return client;
  };
  window.axios = { create };
})();
"""


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_server(url: str, process: subprocess.Popen, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    last_error = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Uvicorn exited early with status {process.returncode}.")
        try:
            with urlopen(url, timeout=1) as response:
                if response.status == 200:
                    return
        except (URLError, TimeoutError, OSError) as exc:
            last_error = exc
        time.sleep(0.25)
    raise RuntimeError(f"ST-NLP test server did not become ready: {last_error}")


def _route_browser_assets(route, base_url: str) -> None:
    url = route.request.url
    if "cdn.jsdelivr.net/npm/axios@" in url:
        route.fulfill(
            status=200,
            content_type="application/javascript",
            body=AXIOS_TEST_ADAPTER,
        )
    elif url.startswith(base_url):
        route.continue_()
    else:
        # Google Identity, Leaflet tiles and fonts are external services. They are
        # deliberately isolated; auth session restoration and both portals are local.
        route.abort()


@pytest.mark.e2e
def test_citizen_submits_report_and_staff_confirms_prediction(tmp_path, monkeypatch):
    database_url = os.getenv("STNLP_E2E_DATABASE_URL")
    if not database_url:
        database_url = f"sqlite+pysqlite:///{(tmp_path / 'stnlp-e2e.sqlite3').as_posix()}"

    secret = "phase24-browser-e2e-secret-" + "x" * 24
    monkeypatch.setattr(settings, "jwt_secret_key", secret)
    monkeypatch.setattr(settings, "jwt_algorithm", "HS256")
    monkeypatch.setattr(settings, "jwt_issuer", "st-nlp-api")
    monkeypatch.setattr(settings, "jwt_audience", "st-nlp-client")
    monkeypatch.setattr(settings, "jwt_access_token_expire_minutes", 30)

    connect_args = {"check_same_thread": False} if database_url.startswith("sqlite") else {}
    engine = create_engine(database_url, connect_args=connect_args, pool_pre_ping=True)
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)

    citizen = User(
        email="browser-citizen@example.com",
        full_name="Browser Citizen",
        role="CITIZEN",
        google_sub=f"google-{uuid4()}",
    )
    staff = User(
        email="browser-receiver@example.com",
        full_name="Browser Receiver",
        role="RECEIVER",
        password_hash=hash_password("Browser-e2e-password-2026!"),
    )
    with Session(engine) as db:
        categories = [
            IncidentCategory(
                code=code,
                name=name,
                slug=slug,
                description=description,
                sort_order=order,
                is_active=True,
            )
            for code, name, slug, description, order in CATEGORIES
        ]
        db.add_all([*categories, citizen, staff])
        db.commit()
        db.refresh(citizen)

    citizen_token, _ = create_access_token(citizen.id, citizen.role)
    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    env = os.environ.copy()
    env.update({
        "DATABASE_URL": database_url,
        "JWT_SECRET_KEY": secret,
        "JWT_ALGORITHM": "HS256",
        "JWT_ISSUER": "st-nlp-api",
        "JWT_AUDIENCE": "st-nlp-client",
        "REPORT_UPLOAD_DIR": str(uploads),
        "PREDICTION_BUNDLE_PATH": str(tmp_path / "intentionally-missing-bundle.joblib"),
        "PYTHONPATH": os.pathsep.join([
            str(PROJECT_ROOT / "backend"),
            str(PROJECT_ROOT / "ml"),
            env.get("PYTHONPATH", ""),
        ]),
    })
    process = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn", "app.main:app",
            "--app-dir", str(PROJECT_ROOT / "backend"),
            "--host", "127.0.0.1", "--port", str(port), "--log-level", "warning",
        ],
        cwd=PROJECT_ROOT,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        _wait_for_server(base_url + "/api/v1/health", process)
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            try:
                citizen_context = browser.new_context()
                citizen_context.route("**/*", lambda route: _route_browser_assets(route, base_url))
                citizen_page = citizen_context.new_page()
                citizen_page.goto(base_url + "/citizen/", wait_until="domcontentloaded")
                citizen_page.evaluate(
                    """(session) => {
                      sessionStorage.setItem("stnlp.citizen.access_token", session.access_token);
                      sessionStorage.setItem("stnlp.citizen.user", JSON.stringify(session.user));
                    }""",
                    {
                        "access_token": citizen_token,
                        "user": {
                            "id": str(citizen.id),
                            "email": citizen.email,
                            "full_name": citizen.full_name,
                            "role": citizen.role,
                        },
                    },
                )
                citizen_page.reload(wait_until="domcontentloaded")
                expect(citizen_page.locator("#auth-notice")).to_contain_text(
                    "Xin chào", timeout=10000,
                )
                expect(citizen_page.locator("#content")).to_be_enabled(timeout=10000)
                citizen_page.locator("#content").fill(
                    "Đường Nguyễn Huệ có ổ gà sâu, gây nguy hiểm cho người đi xe máy."
                )
                citizen_page.locator("#address").fill("Đường Nguyễn Huệ")
                citizen_page.locator("#submit").click()
                expect(citizen_page.locator("#result")).to_be_visible(timeout=15000)
                expect(citizen_page.locator("#result")).to_contain_text(
                    "Hệ thống đã cấp mã theo dõi"
                )
                tracking_code = citizen_page.locator("#result .tracking").inner_text()
                assert tracking_code.startswith("RPT-")

                # No trained artifact is deployed in this isolated UI test, so the citizen
                # portal must show the persisted-report fallback instead of inventing a result.
                with Session(engine) as db:
                    report = db.scalar(select(Report).where(Report.tracking_code == tracking_code))
                    assert report is not None
                    category = db.scalar(
                        select(IncidentCategory).where(IncidentCategory.code == "R02")
                    )
                    model_version = ModelVersion(
                        name="ST-NLP E2E fixture",
                        version="e2e-v1",
                        algorithm="NAIVE_BAYES",
                        representation="TF_IDF",
                        artifact_path="tests/e2e/fixture-bundle.joblib",
                        parameters={"fixture": True},
                        is_active=True,
                    )
                    db.add(model_version)
                    db.flush()
                    db.add(
                        Prediction(
                            report_id=report.id,
                            model_version_id=model_version.id,
                            predicted_category_id=category.id,
                            confidence=0.91,
                            confidence_type="model_score",
                            needs_review=True,
                            review_reasons=["E2E seeded prediction requires staff confirmation."],
                        )
                    )
                    db.commit()
                    report_id = report.id

                citizen_context.close()
                staff_context = browser.new_context()
                staff_context.route("**/*", lambda route: _route_browser_assets(route, base_url))
                staff_page = staff_context.new_page()
                staff_page.goto(base_url + "/admin/", wait_until="domcontentloaded")
                staff_page.locator("#staff-email").fill(staff.email)
                staff_page.locator("#staff-password").fill("Browser-e2e-password-2026!")
                staff_page.locator("#login-button").click()
                expect(staff_page.locator("#admin-app")).to_be_visible(timeout=15000)
                staff_page.locator('[data-view="reports-view"]').click()

                report_button = staff_page.locator(
                    f'#reports-table [data-open-report="{report_id}"]'
                )
                expect(report_button).to_be_visible(timeout=10000)
                report_button.click()
                expect(staff_page.locator("#detail-overlay")).to_be_visible(timeout=10000)
                expect(staff_page.locator("#detail-content")).to_contain_text(tracking_code)
                staff_page.locator("#review-category").select_option("R02")
                staff_page.locator("#review-notes").fill("Đã xác minh hiện trường trong E2E.")
                staff_page.locator("#save-review").click()
                expect(staff_page.locator("#detail-content .history-entry")).to_contain_text(
                    "Đã xác nhận", timeout=15000,
                )
                staff_context.close()
            finally:
                browser.close()

        with Session(engine) as db:
            saved_report = db.scalar(select(Report).where(Report.tracking_code == tracking_code))
            assert saved_report is not None
            assert saved_report.final_category is not None
            assert saved_report.final_category.code == "R02"
            assert len(saved_report.predictions[0].reviews) == 1
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        if database_url.startswith("sqlite"):
            Base.metadata.drop_all(engine)
        engine.dispose()
