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


def main():
    """Entry point for direct execution."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
