"""
SQL Surgeon — Environment Implementation.

Core OpenEnv environment that exposes step(), reset(), and state() for
SQL query optimization tasks.
"""

from typing import Any, Optional
from uuid import uuid4

try:
    from openenv.core.env_server.types import Action, Observation, State
    from openenv.core.env_server import Environment
except Exception:
    # Local fallback so tests can run when openenv-core import path is shadowed.
    from pydantic import BaseModel

    class Action(BaseModel):
        pass

    class Observation(BaseModel):
        done: bool = False
        reward: float = 0.0
        metadata: dict = {}

    class State(BaseModel):
        pass

    class Environment:
        pass

from .database import DatabaseManager, SCHEMA_DDL
from .graders import grade_query, GradeResult
from .tasks import ALL_TASKS, TASK_IDS, Task

# Import our typed models
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import SqlSurgeonAction, SqlSurgeonObservation, SqlSurgeonState


class SqlSurgeonEnvironment(Environment):
    """
    SQL Query Optimization Environment.

    The agent receives a slow SQL query along with the database schema,
    sample data, and query plan. It must rewrite the query to produce
    identical results but execute faster.

    Episode flow:
        1. reset(task_id=...) → loads task, returns schema + slow query
        2. step(optimized_query) → grades the attempt, returns feedback
        3. Repeat step() up to max_attempts or until done
    """

    MAX_ATTEMPTS = 5

    def __init__(self) -> None:
        super().__init__()
        self._db = DatabaseManager(seed=42)
        self._db.initialize()
        self._state = SqlSurgeonState(
            episode_id=str(uuid4()),
            step_count=0,
            task_id="",
            actions_used=0,
            max_actions=15,
            best_speedup=0.0,
            cumulative_reward=0.0,
            best_query="",
            hallucination_count=0,
            total_confidence_error=0.0,
            tool_calls_count=0,
            deception_ignored=False,
            attempt_history=[],
        )
        self._current_task: Optional[Task] = None
        self._original_result = None

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Observation:
        """
        Reset the environment and load deceptive context.
        """
        self._db.reset()

        task_id = kwargs.get("task_id", "filter_scan")
        if task_id not in ALL_TASKS:
            task_id = "filter_scan"
        self._current_task = ALL_TASKS[task_id]

        self._original_result = self._db.execute_query(
            self._current_task.slow_query, timeout_seconds=30.0
        )
        query_plan = self._db.get_query_plan(self._current_task.slow_query)
        sample_data = self._db.get_sample_data(limit=5)

        # Initialize research-grade state
        self._state = SqlSurgeonState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            task_id=task_id,
            actions_used=0,
            max_actions=15,
            best_speedup=0.0,
            cumulative_reward=0.0,
            best_query="",
            hallucination_count=0,
            total_confidence_error=0.0,
            tool_calls_count=0,
            deception_ignored=False,
            attempt_history=[],
        )

        return Observation(
            done=False,
            reward=0.0,
            metadata={
                "task_id": task_id,
                "task_description": self._current_task.description,
                "task_difficulty": self._current_task.difficulty,
                "task_title": self._current_task.title,
                "original_query": self._current_task.slow_query,
                "schema_ddl": SCHEMA_DDL,
                "sample_data": sample_data,
                "query_plan_original": query_plan,
                "deceptive_hints": self._current_task.deceptive_signals,
                "execution_time_original_ms": round(self._original_result.execution_time_ms, 3),
                "expected_row_count": self._original_result.row_count,
                "actions_remaining": 15,
                "status": "ready",
            },
        )

    def step(
        self,
        action: Any,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        """
        Execute an action (Tool call or Submission).
        """
        self._state.step_count += 1
        self._state.actions_used += 1
        
        # Parse action
        if isinstance(action, dict):
            from models import SqlSurgeonActionType
            a_type = action.get("action_type", "submit")
            query = action.get("query", "")
            confidence = action.get("confidence", 1.0)
            thoughts = action.get("thoughts", "")
        else:
            a_type = getattr(action, "action_type", "submit")
            query = getattr(action, "query", "")
            confidence = getattr(action, "confidence", 1.0)
            thoughts = getattr(action, "thoughts", "")

        if not self._current_task:
            return Observation(done=True, reward=0.0, metadata={"error": "No task loaded"})

        # Branch based on Action Type
        from models import SqlSurgeonActionType
        
        tool_result = None
        reward = 0.0
        done = False
        metadata = {
            "task_id": self._state.task_id,
            "actions_remaining": self._state.max_actions - self._state.actions_used,
        }

        if a_type == SqlSurgeonActionType.CHECK_SCHEMA:
            self._state.tool_calls_count += 1
            tool_result = self._db.get_schema_info()
            reward = 0.0
            
        elif a_type == SqlSurgeonActionType.RUN_EXPLAIN:
            self._state.tool_calls_count += 1
            tool_result = self._db.get_query_plan(query)
            reward = 0.0
            
        elif a_type == SqlSurgeonActionType.THINK:
            tool_result = f"Reasoning logged: {len(thoughts)} characters."
            reward = 0.0
            
        elif a_type == SqlSurgeonActionType.SUBMIT:
            grade = grade_query(
                db=self._db,
                original_query=self._current_task.slow_query,
                optimized_query=query,
                confidence=confidence,
                deceptive_signals=self._current_task.deceptive_signals
            )
            
            # Update Metrics
            if not grade.is_correct and grade.reward < 0:
                self._state.hallucination_count += 1
            
            self._state.total_confidence_error += grade.confidence_error
            self._state.deception_ignored = grade.deception_resistance > 0
            
            reward = grade.reward
            done = True
            metadata.update({
                "is_correct": grade.is_correct,
                "speedup": grade.speedup,
                "hallucination_info": grade.hallucination_info,
                "error": grade.error,
                "confidence_error": grade.confidence_error,
                "deception_resistance": grade.deception_resistance,
            })

        # Check budget
        if self._state.actions_used >= self._state.max_actions:
            done = True

        self._state.cumulative_reward += reward
        metadata["tool_result"] = tool_result
        metadata["cumulative_reward"] = round(self._state.cumulative_reward, 3)

        return Observation(
            done=done,
            reward=reward,
            metadata=metadata
        )

    @property
    def state(self) -> State:
        """Get current environment state."""
        return self._state
