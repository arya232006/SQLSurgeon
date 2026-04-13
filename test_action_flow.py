#!/usr/bin/env python
"""Integration test to verify the action flow works end-to-end."""

import asyncio
import json
from models import SqlSurgeonAction, SqlSurgeonActionType
from server.sql_surgeon_environment import SqlSurgeonEnvironment

async def test_action_flow():
    """Test that actions flow correctly through the environment."""
    env = SqlSurgeonEnvironment()
    
    # Reset with a task
    obs = env.reset(task_id="filter_scan")
    print("✓ Environment reset")
    print(f"  Task: {obs.task_id}")
    print(f"  Description: {obs.task_description[:100]}...")
    print()
    
    # Test SCHEMA action
    print("Testing SCHEMA action...")
    schema_action = SqlSurgeonAction(
        action_type=SqlSurgeonActionType.CHECK_SCHEMA,
        query="",
        thoughts="Getting schema"
    )
    obs = env.step(schema_action)
    tool_result = obs.metadata.get("tool_result", "")
    assert tool_result, "Schema result is empty!"
    print(f"✓ Got schema ({len(tool_result)} chars)")
    print(f"  Sample: {tool_result[:200]}...")
    print()
    
    # Test EXPLAIN action with original query
    print("Testing EXPLAIN action with original query...")
    original_query = obs.original_query
    explain_action = SqlSurgeonAction(
        action_type=SqlSurgeonActionType.RUN_EXPLAIN,
        query=original_query,
        thoughts="Analyzing original query plan"
    )
    obs = env.step(explain_action)
    plan_result = obs.metadata.get("tool_result", "")
    assert plan_result, "Query plan is empty!"
    print(f"✓ Got query plan ({len(plan_result)} chars)")
    print(f"  Sample: {plan_result[:200]}...")
    print()
    
    # Test THINK action
    print("Testing THINK action...")
    think_action = SqlSurgeonAction(
        action_type=SqlSurgeonActionType.THINK,
        query="",
        thoughts="I should add an index on the join column"
    )
    obs = env.step(think_action)
    think_result = obs.metadata.get("tool_result", "")
    print(f"✓ Think logged: {think_result}")
    print()
    
    # Test SUBMIT action (will fail because we haven't optimized)
    print("Testing SUBMIT action...")
    submit_action = SqlSurgeonAction(
        action_type=SqlSurgeonActionType.SUBMIT,
        query=original_query,  # Submit the same query
        confidence=0.5,
        thoughts="Submitting unoptimized query as test"
    )
    obs = env.step(submit_action)
    assert obs.done, "Submit should mark episode as done"
    is_correct = obs.metadata.get("is_correct", False)
    speedup = obs.metadata.get("speedup", 0.0)
    print(f"✓ Submit received (is_correct={is_correct}, speedup={speedup}x)")
    print()
    
    print("All tests passed!")

if __name__ == "__main__":
    asyncio.run(test_action_flow())
