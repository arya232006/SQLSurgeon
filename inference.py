"""
SQL Surgeon — Baseline Inference Script
========================================

ENVIRONMENT VARIABLES (submission checklist):
    API_BASE_URL         LLM API endpoint — default set in code (optional override)
    MODEL_NAME           Model id — default set in code (optional override)
    HF_TOKEN             Hugging Face / API key — REQUIRED, no default (set in environment)
    LOCAL_IMAGE_NAME     Optional — local Docker image when using from_docker_image()
    INFERENCE_DEBUG      Optional — set to 1 to log LLM parse failures and empty responses
    INFERENCE_JSON_MODE  Optional — set to 1 to request JSON object mode (OpenAI-compatible)

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
INFERENCE_DEBUG = os.getenv("INFERENCE_DEBUG", "").strip().lower() in ("1", "true", "yes")
INFERENCE_JSON_MODE = os.getenv("INFERENCE_JSON_MODE", "").strip().lower() in ("1", "true", "yes")

BENCHMARK = "sql_surgeon"
MAX_ACTIONS = 15 # Budget for tools + submission
TEMPERATURE = 0.1
FORCE_JSON_MODE = True  # Always use JSON response format to prevent LLM from reasoning in prose


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

SYSTEM_PROMPT = """You must output ONLY valid JSON. No markdown, no prose, no explanations.

Your task: Optimize a slow SQL query to make it faster while keeping results identical.

FIRST ACTION (mandatory):
{"action_type": "schema"}

Then follow this workflow:
1. After schema response, use "explain" to test queries
2. Verify your optimization doesn't change results
3. Submit the optimized query with confidence score

AVAILABLE ACTIONS (respond with ONE per turn):
1. schema - Get database tables, columns, indexes
   {"action_type": "schema"}

2. explain - Analyze query performance
   {"action_type": "explain", "query": "SELECT ..."}

3. think - Internal reasoning
   {"action_type": "think", "thoughts": "your analysis"}

4. submit - Final optimized query
   {"action_type": "submit", "query": "SELECT ...", "confidence": 0.95}

CRITICAL RULES:
- Respond with ONLY a JSON object
- No nested objects except what's shown above
- action_type must be one of: schema, explain, think, submit
- confidence must be 0.0 to 1.0
- First response MUST have action_type = "schema"
- Test optimizations with explain before submitting
- Never submit without testing first

RESPONSE MUST BE VALID JSON ONLY. No other text.
"""


def _completion_text(message: Any) -> str:
    """Normalize `message.content` from the OpenAI-compatible API (str or content parts)."""
    raw = getattr(message, "content", None)
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):
        chunks: List[str] = []
        for part in raw:
            if isinstance(part, dict):
                t = part.get("text")
                if isinstance(t, str):
                    chunks.append(t)
                elif isinstance(part.get("content"), str):
                    chunks.append(part["content"])
            elif isinstance(part, str):
                chunks.append(part)
        return "".join(chunks)
    return str(raw)


def _reasoning_extras(message: Any) -> str:
    """Some HF / reasoning models put the visible answer in separate fields."""
    bits: List[str] = []
    for attr in (
        "reasoning",
        "reasoning_content",
        "reasoningContent",
        "thinking",
    ):
        v = getattr(message, attr, None)
        if isinstance(v, str) and v.strip():
            bits.append(v.strip())
    return "\n".join(bits)


def _all_assistant_text(message: Any) -> str:
    """Content + reasoning fields (order: main content first, then reasoning tails)."""
    main = _completion_text(message).strip()
    extra = _reasoning_extras(message).strip()
    if main and extra:
        return f"{main}\n{extra}"
    return main or extra


def _strip_thinking_markers(text: str) -> str:
    """Drop common reasoning wrappers so JSON scanners see the action object."""
    t = text
    t = re.sub(r"<redacted_thinking>[\s\S]*?</redacted_thinking>", "", t, flags=re.IGNORECASE)
    _think_o = chr(60) + "think" + chr(62)
    _think_c = "</" + "think" + ">"
    t = re.sub(
        re.escape(_think_o) + r"[\s\S]*?" + re.escape(_think_c),
        "",
        t,
        flags=re.IGNORECASE,
    )
    t = re.sub(r"```\s*thinking\s*[\s\S]*?```", "", t, flags=re.IGNORECASE)
    return t.strip()


def _after_last_closer(text: str, closer: str) -> Optional[str]:
    idx = text.lower().rfind(closer.lower())
    if idx < 0:
        return None
    return text[idx + len(closer) :].strip()


def _parse_candidates(text: str) -> List[str]:
    """Try the full message, then tails after reasoning closers (JSON is usually last)."""
    text = (text or "").strip()
    if not text:
        return []
    seen: List[str] = []

    def add(s: str) -> None:
        s = (s or "").strip()
        if s and s not in seen:
            seen.append(s)

    add(text)
    add(_strip_thinking_markers(text))
    for closer in (
        "</redacted_thinking>",
        "</" + "think" + ">",
        "`</redacted_thinking>`",
        "`</redacted_thinking>`",
    ):
        tail = _after_last_closer(text, closer)
        if tail:
            add(_strip_thinking_markers(tail))
    return seen


def _normalize_parsed_action(parsed: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Map varied LLM JSON shapes to our SqlSurgeonAction fields.

    Models often emit camelCase (`actionType`), nest under `action`, or use synonyms.
    """
    if not isinstance(parsed, dict):
        return None

    merged: Dict[str, Any] = {**parsed}
    inner = merged.get("action")
    if isinstance(inner, dict):
        merged = {**inner, **{k: v for k, v in merged.items() if k != "action"}}

    raw_type = (
        merged.get("action_type")
        or merged.get("actionType")
        or merged.get("tool")
        or merged.get("type")
    )
    if raw_type is None:
        return None
    if isinstance(raw_type, (list, tuple)) and raw_type:
        raw_type = raw_type[0]
    if not isinstance(raw_type, str):
        return None

    t = raw_type.strip().lower().replace("-", "_")
    aliases = {
        "run_explain": "explain",
        "explain_query": "explain",
        "check_schema": "schema",
        "get_schema": "schema",
        "schema_tool": "schema",
        "reasoning": "think",
        "reason": "think",
        "reflection": "think",
        "final_submit": "submit",
        "answer": "submit",
    }
    t = aliases.get(t, t)
    merged["action_type"] = t

    # Common alternate field names for SQL text
    if not merged.get("query"):
        for key in ("sql", "optimized_query", "optimizedQuery", "statement"):
            v = merged.get(key)
            if isinstance(v, str) and v.strip():
                merged["query"] = v
                break

    return merged


def _parse_action_segment(text: str) -> Optional[Dict[str, Any]]:
    """Try to pull one JSON action object from a single candidate string."""
    text = (text or "").strip()
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

    stripped = text.strip()
    if stripped.startswith("{") or stripped.startswith("["):
        try:
            blob = json.loads(stripped)
            if isinstance(blob, list):
                for item in blob:
                    if isinstance(item, dict):
                        norm = _normalize_parsed_action(item)
                        if norm is not None:
                            return norm
                return None
            if isinstance(blob, dict):
                return blob
        except json.JSONDecodeError:
            pass
    return None


def _parse_action_from_llm_text(content: str) -> Optional[Dict[str, Any]]:
    """
    Extract a JSON action object from model output.

    Tries stripped text, tails after reasoning tags, and each candidate segment.
    """
    for cand in _parse_candidates(content or ""):
        got = _parse_action_segment(cand)
        if got is not None:
            return got
    return None


async def get_next_action(client: OpenAI, history: List[Dict]) -> Dict:
    """Call the LLM to get the next action in the multi-turn loop."""
    try:
        req: Dict[str, Any] = {
            "model": MODEL_NAME,
            "messages": history,
            "temperature": TEMPERATURE,
            "max_tokens": 2048,
        }
        if INFERENCE_DEBUG:
            print(f"[DEBUG] Calling LLM with {len(history)} messages, last user prompt: {history[-1]['content'][:100] if history else '(no history)'}...", flush=True)
        if INFERENCE_JSON_MODE or FORCE_JSON_MODE:
            req["response_format"] = {"type": "json_object"}
        completion = client.chat.completions.create(**req)
        message = completion.choices[0].message
        full_raw = _all_assistant_text(message)
        
        # ALWAYS log the LLM response for debugging
        if not full_raw.strip():
            fr = getattr(completion.choices[0], "finish_reason", None)
            print(f"[DEBUG] Empty LLM text finish_reason={fr!r}", flush=True)
        else:
            preview = full_raw[:300].replace("\n", "\\n")
            print(f"[DEBUG] LLM response: {preview!r}...", flush=True)

        parsed = _parse_action_from_llm_text(full_raw)
        if isinstance(parsed, dict):
            normalized = _normalize_parsed_action(parsed)
            if normalized is not None:
                at = normalized.get("action_type")
                print(
                    f"[DEBUG] Parsed OK: action_type={at!r} "
                    f"query_len={len(str(normalized.get('query') or ''))}",
                    flush=True,
                )
                return normalized

        print(f"[DEBUG] Parse FAILED - returning default think action", flush=True)
        preview = full_raw[:1200].replace("\n", "\\n") if full_raw else "(empty)"
        print(f"[DEBUG] Full LLM text: {preview!r}", flush=True)


        return {
            "action_type": "think",
            "thoughts": f"Parsing error in LLM output: {full_raw[:500]}",
        }
    except Exception as e:
        return {"action_type": "think", "thoughts": f"Inference Error: {e}"}


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
            obs_dict = obs.metadata if isinstance(obs.metadata, dict) else {}
            initial_context = {
                "task_id": obs_dict.get("task_id", task_id),
                "description": obs_dict.get("task_description", ""),
                "original_query": obs_dict.get("original_query", ""),
                "actions_remaining": obs_dict.get("actions_remaining", MAX_ACTIONS),
            }
            prompt = f"TASK LOADED: {task_id}\n\n{json.dumps(initial_context, indent=2)}"

            for step in range(1, MAX_ACTIONS + 1):
                if result.done:
                    break
                steps_taken = step
                history.append({"role": "user", "content": prompt})

                action_dict = await get_next_action(client, history)
                action_type_str = action_dict.get("action_type", "think").lower().strip()
                
                # Map string to enum
                action_enum = SqlSurgeonActionType.THINK
                for enum_val in SqlSurgeonActionType:
                    if enum_val.value == action_type_str:
                        action_enum = enum_val
                        break
                
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
                metadata = obs.metadata if isinstance(obs.metadata, dict) else {}

                action_str = action_dict.get("query") or action_enum.value
                action_str = str(action_str).replace("\n", " ").strip()[:200]
                error_val = metadata.get("error")
                log_step(step=step, action=action_str, reward=reward, done=done, error=error_val)
                rewards.append(reward)

                if action_enum == SqlSurgeonActionType.SUBMIT:
                    result_summary = {
                        "status": "submitted",
                        "is_correct": metadata.get("is_correct", False),
                        "speedup": metadata.get("speedup", 0.0),
                        "error": metadata.get("error"),
                    }
                    prompt = f"Submission result:\n{json.dumps(result_summary, indent=2)}"
                else:
                    tool_result = metadata.get("tool_result") or metadata.get("error") or "No output"
                    result_info = {
                        "tool": action_enum.value,
                        "result": str(tool_result)[:500],
                    }
                    prompt = f"Tool result:\n{json.dumps(result_info, indent=2)}"

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
