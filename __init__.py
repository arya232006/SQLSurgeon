"""
SQL Surgeon — package exports.

Exports are resolved lazily to avoid import-time failures in tooling (e.g. pytest
collection) when optional runtime dependencies are not yet available.
"""

from typing import Any

__all__ = ["SqlSurgeonEnv", "SqlSurgeonAction", "SqlSurgeonActionType", "SqlSurgeonObservation", "SqlSurgeonState"]


def __getattr__(name: str) -> Any:
    if name in {"SqlSurgeonAction", "SqlSurgeonActionType", "SqlSurgeonObservation", "SqlSurgeonState"}:
        from .models import (
            SqlSurgeonAction,
            SqlSurgeonActionType,
            SqlSurgeonObservation,
            SqlSurgeonState,
        )
        exports = {
            "SqlSurgeonAction": SqlSurgeonAction,
            "SqlSurgeonActionType": SqlSurgeonActionType,
            "SqlSurgeonObservation": SqlSurgeonObservation,
            "SqlSurgeonState": SqlSurgeonState,
        }
        return exports[name]
    if name == "SqlSurgeonEnv":
        from .client import SqlSurgeonEnv
        return SqlSurgeonEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
