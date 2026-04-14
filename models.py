"""
SQL Surgeon — Typed Models.
"""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import Field

try:
    from openenv.core.env_server.types import Action, Observation, State
except Exception:
    # Local fallback so tests can run even if openenv-core is shadowed.
    from pydantic import BaseModel

    class Action(BaseModel):
        pass

    class Observation(BaseModel):
        done: bool = False
        reward: float = 0.0
        metadata: Dict[str, Any] = Field(default_factory=dict)

    class State(BaseModel):
        pass


class SqlSurgeonActionType(str, Enum):
    """Available action types for the agent."""
    SUBMIT = "submit"          # Final submission for grading
    RUN_EXPLAIN = "explain"    # Tool: Get execution plan
    CHECK_SCHEMA = "schema"    # Tool: Get detailed schema info
    THINK = "think"            # Tool: Internal reasoning (logged)


class SqlSurgeonAction(Action):
    action_type: SqlSurgeonActionType = SqlSurgeonActionType.SUBMIT
    query: str = ""
    thoughts: str = ""
    confidence: float = 1.0


class SqlSurgeonObservation(Observation):
    task_id: str = ""
    task_description: str = ""
    original_query: str = ""
    schema_ddl: str = ""
    sample_data: str = ""
    query_plan_original: str = ""
    execution_time_original_ms: float = 0.0
    expected_row_count: int = 0
    deceptive_hints: List[str] = Field(default_factory=list)

    tool_result: Optional[str] = None
    last_error: Optional[str] = None
    last_execution_time_ms: float = 0.0
    last_speedup: float = 0.0
    last_correctness: bool = False
    hallucination_info: Optional[Dict[str, Any]] = None
    actions_remaining: int = 0
    attempts_remaining: int = 0  # backward-compatible alias
    best_speedup_so_far: float = 0.0


class SqlSurgeonState(State):
    episode_id: str = ""
    step_count: int = 0
    task_id: str = ""
    actions_used: int = 0
    max_actions: int = 15
    best_speedup: float = 0.0
    cumulative_reward: float = 0.0
    best_query: str = ""
    hallucination_count: int = 0
    total_confidence_error: float = 0.0
    tool_calls_count: int = 0
    deception_ignored: bool = False
    attempt_history: List[str] = Field(default_factory=list)
