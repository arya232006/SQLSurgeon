"""
FastAPI application for the SQL Surgeon Environment.

Usage:
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
"""

try:
    from openenv.core.env_server.http_server import create_app
    from openenv.core.env_server.types import Action, Observation
    from .sql_surgeon_environment import SqlSurgeonEnvironment
except ImportError:
    from openenv.core.env_server.http_server import create_app
    from openenv.core.env_server.types import Action, Observation
    from server.sql_surgeon_environment import SqlSurgeonEnvironment

# Create the app — pass the class for WebSocket session support
app = create_app(
    SqlSurgeonEnvironment,
    Action,
    Observation,
    env_name="sql_surgeon",
)


def main():
    """Entry point for direct execution."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
