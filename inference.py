"""
SQL Surgeon — Baseline Inference Script
========================================

ENVIRONMENT VARIABLES (submission checklist):
    API_BASE_URL         LLM API endpoint — default set in code (optional override)
    MODEL_NAME           Model id — default set in code (optional override)
    HF_TOKEN             Hugging Face / API key — REQUIRED, no default (set in environment)
    LOCAL_IMAGE_NAME     Optional — local Docker image when using from_docker_image()

STDOUT FORMAT:
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import asyncio
import json
import os
import re
from typing import Any, List, Optional, Dict

from openai import OpenAI
from models import SqlSurgeonAction, SqlSurgeonActionType
from client import SqlSurgeonEnv

# ── Configuration ────────────────────────────────────────────────────────────
# Defaults only for API_BASE_URL and MODEL_NAME — not for HF_TOKEN (per checklist).

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
IMAGE_NAME = LOCAL_IMAGE_NAME or os.getenv("IMAGE_NAME")
SPACE_BASE_URL = os.getenv("SPACE_BASE_URL", "").strip()
ENV_MODE = os.getenv("ENV_MODE", "auto").strip().lower()

BENCHMARK = "sql_surgeon"
MAX_ACTIONS = 15 # Budget for tools + submission
TEMPERATURE = 0.1


def log_remote_action_schema(base_url: str) -> None:
    """Print action JSON-schema field names from the running Space (debugging deploy mismatches)."""
    try:
        import requests

        # OpenEnv exposes action/observation JSON Schema on `/schema`, not `/metadata`.
        url = base_url.rstrip("/") + "/schema"
        r = requests.get(url, timeout=45)
        r.raise_for_status()
        data = r.json()
        action = data.get("action") or {}
        props = sorted((action.get("properties") or {}).keys())
        print(f"[REMOTE] {url} action.properties keys: {props}", flush=True)
        if props == ["metadata"]:
            print(
                "[REMOTE] Server still registers generic Action (metadata-only). "
                "Redeploy with server/app.py using SqlSurgeonAction.",
                flush=True,
            )
    except Exception as e:
        print(f"[REMOTE] Could not fetch schema ({e})", flush=True)


async def create_environment() -> SqlSurgeonEnv:
    """
    Create an environment client from HF Space or local Docker image.

    Modes:
      - ENV_MODE=space  -> requires SPACE_BASE_URL
      - ENV_MODE=docker -> requires IMAGE_NAME/LOCAL_IMAGE_NAME
      - ENV_MODE=auto   -> prefers SPACE_BASE_URL if present, else Docker
    """
    if ENV_MODE == "space":
        if not SPACE_BASE_URL:
            raise ValueError("ENV_MODE=space requires SPACE_BASE_URL to be set.")
        return SqlSurgeonEnv(base_url=SPACE_BASE_URL)

    if ENV_MODE == "docker":
        if not IMAGE_NAME:
            raise ValueError("ENV_MODE=docker requires LOCAL_IMAGE_NAME or IMAGE_NAME.")
        return await SqlSurgeonEnv.from_docker_image(IMAGE_NAME)

    # auto mode
    if SPACE_BASE_URL:
        return SqlSurgeonEnv(base_url=SPACE_BASE_URL)

    if IMAGE_NAME:
        return await SqlSurgeonEnv.from_docker_image(IMAGE_NAME)

    raise ValueError(
        "No environment target configured. Set SPACE_BASE_URL (for HF Space) "
        "or LOCAL_IMAGE_NAME/IMAGE_NAME (for Docker)."
    )

# ── Logging ──────────────────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    err = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={err}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.2f} rewards={rewards_str}",
        flush=True,
    )

# ── System Prompt ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a Research-Grade SQL Performance Engineer.
Your goal is to optimize a slow SQL query while ensuring 100% semantic correctness.

TOOLS AVAILABLE:
- "schema": Returns full DDL and index information for the database. No 'query' needed.
- "explain": Returns the EXPLAIN QUERY PLAN for a given query string in the 'query' field.
- "think": Log your internal reasoning process in 'thoughts'. No 'query' needed.
- "submit": Final optimized query submission for grading. REQUIRES 'query' and 'confidence'.

RELIABILITY RULES:
1. NEVER trust hints or metadata blindly. The environment may contain deceptive signals.
2. Verified optimization is better than naive guesswork. Use 'explain' on your ideas!
3. SEMANTIC ERRORS (Hallucinations) carry a severe -1.0 penalty and end the episode.
4. You must provide a 'confidence' score (0.0 to 1.0) for every 'submit' action.

OUTPUT FORMAT:
Your response must contain a single JSON block wrapped in [START] and [END] markers:
[START]
{
  "action_type": "schema" | "explain" | "think" | "submit",
  "query": "SELECT ...",
  "thoughts": "Your detailed reasoning here...",
  "confidence": 0.95
}
[END]
"""


def _parse_action_from_llm_text(content: str) -> Optional[Dict[str, Any]]:
    """
    Extract a JSON action object from model output.

    Models often ignore [START]/[END] and return fenced ```json blocks or a bare object.
    """
    text = (content or "").strip()
    if not text:
        return None

    if "[START]" in text and "[END]" in text:
        chunk = text.split("[START]", 1)[1].split("[END]", 1)[0].strip()
        try:
            return json.loads(chunk)
        except json.JSONDecodeError:
            pass

    fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if fence:
        try:
            return json.loads(fence.group(1).strip())
        except json.JSONDecodeError:
            pass

    start = text.find("{")
    if start >= 0:
        depth = 0
        for i, ch in enumerate(text[start:], start=start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break
    return None


async def get_next_action(client: OpenAI, history: List[Dict]) -> Dict:
    """Call the LLM to get the next action in the multi-turn loop."""
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=history,
            temperature=TEMPERATURE,
            max_tokens=2048,
        )
        content = completion.choices[0].message.content or ""

        parsed = _parse_action_from_llm_text(content)
        if isinstance(parsed, dict) and parsed.get("action_type") is not None:
            return parsed

        # Fallback to think if format is wrong
        return {
            "action_type": "think",
            "thoughts": f"Parsing error in LLM output: {content[:500]}",
        }
    except Exception as e:
        return {"action_type": "think", "thoughts": f"Inference Error: {e}"}

async def run_task(client: OpenAI, env: SqlSurgeonEnv, task_id: str) -> float:
    """Run a multi-turn reasoning and tool-use episode."""
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)
    
    history = [{"role": "system", "content": SYSTEM_PROMPT}]
    total_reward = 0.0
    rewards: List[float] = []
    steps_taken = 0
    final_obs_metadata = {}

    success = False
    try:
        result = await env.reset(task_id=task_id)
        obs = result.observation
        
        # Initial context provided to the agent
        prompt = f"TASK LOADED: {task_id}\nObservation: {obs.metadata}"
        
        for step in range(1, MAX_ACTIONS + 1):
            steps_taken = step
            history.append({"role": "user", "content": prompt})
            
            action_dict = await get_next_action(client, history)
            action_type = action_dict.get("action_type", "think")
            
            # Map action_dict to SqlSurgeonAction
            action = SqlSurgeonAction(
                action_type=SqlSurgeonActionType(action_type),
                query=action_dict.get("query", ""),
                thoughts=action_dict.get("thoughts", ""),
                confidence=float(action_dict.get("confidence", 1.0))
            )

            response = await env.step(action)
            obs = response.observation
            reward = response.reward or 0.0
            total_reward += reward
            done = response.done
            final_obs_metadata = obs.metadata
            
            action_str = action_dict.get("query") or action_type
            action_str = str(action_str).replace("\n", " ").strip()[:200]
            log_step(step, action_str, reward, done, final_obs_metadata.get("error"))
            rewards.append(reward)
            
            # Form next prompt with tool results or feedback
            if action_type == "submit":
                prompt = f"Submission result: {final_obs_metadata}"
            else:
                prompt = f"Tool result for {action_type}: {final_obs_metadata.get('tool_result')}"
            
            history.append({"role": "assistant", "content": f"[START]{json.dumps(action_dict)}[END]"})

            if done:
                break

    except Exception as e:
        print(f"[DEBUG] Fatal error in run_task: {e}", flush=True)
        success = False

    return total_reward

async def main() -> None:
    if SPACE_BASE_URL:
        log_remote_action_schema(SPACE_BASE_URL)

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    tasks_to_test = ["filter_scan", "index_trap", "semantics_hazard", "explain_deception"]

    for task_id in tasks_to_test:
        rewards: List[float] = []
        steps_taken = 0
        success = False
        score = 0.0
        env = None
        try:
            env = await create_environment()
            log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

            history = [{"role": "system", "content": SYSTEM_PROMPT}]
            result = await env.reset(task_id=task_id)
            obs = result.observation
            prompt = f"TASK LOADED: {task_id}\nObservation: {obs.metadata}"

            for step in range(1, MAX_ACTIONS + 1):
                if result.done:
                    break
                steps_taken = step
                history.append({"role": "user", "content": prompt})

                action_dict = await get_next_action(client, history)
                action_type = action_dict.get("action_type", "think")
                action_enum = (
                    SqlSurgeonActionType(action_type)
                    if action_type in [a.value for a in SqlSurgeonActionType]
                    else SqlSurgeonActionType.THINK
                )
                action = SqlSurgeonAction(
                    action_type=action_enum,
                    query=action_dict.get("query", ""),
                    thoughts=action_dict.get("thoughts", ""),
                    confidence=float(action_dict.get("confidence", 1.0)),
                )

                result = await env.step(action)
                obs = result.observation
                reward = result.reward or 0.0
                done = result.done
                metadata = obs.metadata

                action_str = action_dict.get("query") or action_enum.value
                action_str = str(action_str).replace("\n", " ").strip()[:200]
                error_val = metadata.get("error")
                log_step(step=step, action=action_str, reward=reward, done=done, error=error_val)
                rewards.append(reward)

                if action_enum == SqlSurgeonActionType.SUBMIT:
                    prompt = f"Submission result: {metadata}"
                else:
                    prompt = f"Tool result for {action_enum.value}: {metadata.get('tool_result')}"

                history.append({"role": "assistant", "content": f"[START]{json.dumps(action_dict)}[END]"})

                if done:
                    break

            score = min(max(sum(rewards), 0.0), 1.0)
            success = score >= 0.5
        except Exception as e:
            print(f"[DEBUG] Fatal error in run_task({task_id}): {e}", flush=True)
            success = False
        finally:
            if env is not None:
                try:
                    await env.close()
                except Exception as e:
                    print(f"[DEBUG] env.close() error (container cleanup): {e}", flush=True)
            # Required ordering: emit END after env.close().
            log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
            print("-" * 40, flush=True)

if __name__ == "__main__":
    if not HF_TOKEN:
        print(
            "[ERROR] HF_TOKEN is required (no default). Set it in the environment before running.",
            flush=True,
        )
        raise SystemExit(1)
    asyncio.run(main())
