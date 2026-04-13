"""
SQL Surgeon — Client.

Provides the EnvClient subclass for connecting to a running
SQL Surgeon environment server.
"""

from openenv.core import EnvClient
try:
    # Older OpenEnv versions
    from openenv.core import StepResult
except Exception:
    try:
        # Newer OpenEnv versions
        from openenv.core.types import StepResult
    except Exception:
        # Fallback for type hints only; runtime behavior remains unchanged
        from typing import Any as StepResult
from models import SqlSurgeonAction, SqlSurgeonObservation, SqlSurgeonState


class SqlSurgeonEnv(EnvClient[SqlSurgeonAction, SqlSurgeonObservation, SqlSurgeonState]):
    """
    Client for the SQL Surgeon environment.

    Example (Docker):
        >>> env = SqlSurgeonEnv.from_docker_image("sql-surgeon:latest")
        >>> result = env.reset(task_id="filter_scan")
        >>> print(result.observation)
        >>> result = env.step(SqlSurgeonAction(optimized_query="SELECT ..."))
        >>> print(result.reward)
        >>> env.close()

    Example (HF Space):
        >>> with SqlSurgeonEnv(base_url="https://your-space.hf.space").sync() as env:
        ...     env.reset(task_id="subquery_to_join")
        ...     result = env.step(SqlSurgeonAction(optimized_query="..."))
    """

    def _step_payload(self, action: SqlSurgeonAction) -> dict:
        return {
            "action_type": action.action_type,
            "query": action.query,
            "thoughts": action.thoughts,
            "confidence": action.confidence,
        }

    def _parse_result(self, payload: dict) -> StepResult[SqlSurgeonObservation]:
        # Wire format: { "observation": { ...typed fields... }, "reward", "done" }.
        # `metadata` is omitted from JSON by OpenEnv's serializer; fields live under "observation".
        inner = dict(payload.get("observation") or {})
        extra = payload.get("metadata")
        if isinstance(extra, dict):
            inner = {**inner, **extra}
        dh = inner.get("deceptive_hints", [])
        if not isinstance(dh, list):
            dh = []
        hi = inner.get("hallucination_info")
        if hi is not None and not isinstance(hi, dict):
            hi = None
        obs = SqlSurgeonObservation(
            metadata=inner,
            task_id=str(inner.get("task_id", "")),
            task_description=str(inner.get("task_description", "")),
            original_query=str(inner.get("original_query", "")),
            schema_ddl=str(inner.get("schema_ddl", "")),
            sample_data=str(inner.get("sample_data", "")),
            query_plan_original=str(inner.get("query_plan_original", "")),
            execution_time_original_ms=float(inner.get("execution_time_original_ms", 0.0)),
            expected_row_count=int(inner.get("expected_row_count", 0)),
            deceptive_hints=dh,
            tool_result=inner.get("tool_result"),
            last_error=inner.get("error"),
            last_execution_time_ms=float(inner.get("execution_time_optimized_ms", 0.0)),
            last_speedup=float(inner.get("speedup", 0.0)),
            last_correctness=bool(inner.get("is_correct", False)),
            hallucination_info=hi,
            actions_remaining=int(inner.get("actions_remaining", 0)),
            attempts_remaining=int(inner.get("actions_remaining", 0)),
            best_speedup_so_far=float(inner.get("best_speedup_so_far", 0.0)),
        )
        return StepResult(
            observation=obs,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict) -> SqlSurgeonState:
        return SqlSurgeonState(
            episode_id=payload.get("episode_id", ""),
            step_count=payload.get("step_count", 0),
            task_id=payload.get("task_id", ""),
            actions_used=payload.get("actions_used", 0),
            max_actions=payload.get("max_actions", 15),
            best_speedup=payload.get("best_speedup", 0.0),
            cumulative_reward=payload.get("cumulative_reward", 0.0),
        )
