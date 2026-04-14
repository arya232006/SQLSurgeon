"""
FastAPI application for the SQL Surgeon Environment.

Usage:
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
"""

import os
import sys

# Repo root (parent of server/) so `models` resolves for OpenEnv deserialization.
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from fastapi.openapi.docs import get_redoc_html, get_swagger_ui_html
from fastapi.responses import RedirectResponse

from models import SqlSurgeonAction, SqlSurgeonObservation

try:
    from openenv.core.env_server.http_server import create_app
    from .sql_surgeon_environment import SqlSurgeonEnvironment
except ImportError:
    from openenv.core.env_server.http_server import create_app
    from server.sql_surgeon_environment import SqlSurgeonEnvironment

# Use SqlSurgeonAction / SqlSurgeonObservation so WebSocket step payloads validate.
# Generic openenv Action only allows `metadata` (extra=forbid); tool fields would 422.
app = create_app(
    SqlSurgeonEnvironment,
    SqlSurgeonAction,
    SqlSurgeonObservation,
    env_name="sql_surgeon",
)

# FastAPI's default docs use absolute `/openapi.json`, which breaks when the app is
# served behind a path prefix (for example, HF Spaces embedded proxy routes).
# Re-register docs endpoints with relative spec URLs so they work in both contexts.
app.router.routes = [
    route
    for route in app.router.routes
    if route.path not in {"/docs", "/redoc"}
]


@app.get("/docs", include_in_schema=False)
def custom_swagger_ui() -> object:
    return get_swagger_ui_html(
        openapi_url="openapi.json",
        title=f"{app.title} - Swagger UI",
    )


@app.get("/redoc", include_in_schema=False)
def custom_redoc() -> object:
    return get_redoc_html(
        openapi_url="openapi.json",
        title=f"{app.title} - ReDoc",
    )


@app.get("/")
def _root() -> RedirectResponse:
    # Hugging Face Spaces loads `/`; OpenEnv’s HTTP API has no handler there by default.
    return RedirectResponse(url="docs")


def main():
    """Entry point for direct execution."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
