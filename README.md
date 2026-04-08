---
title: SQL Surgeon
emoji: 🏥
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
---

# 🏥 SQL Surgeon

> **A real-world SQL query optimization environment for OpenEnv.**
> Train AI agents to rewrite slow SQL queries into fast ones — with deterministic grading.

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compatible-blue)](https://github.com/meta-pytorch/OpenEnv)
[![License](https://img.shields.io/badge/license-BSD--3-green)](LICENSE)

---

## 🎯 Motivation

SQL query optimization is one of the most universal, high-value tasks in software engineering. Every company with a database — from startups to FAANG — spends significant engineering hours tuning slow queries. This environment creates a **standardized benchmark** for training and evaluating AI agents on this critical skill.

Unlike toy environments, SQL Surgeon uses:
- A **real SQLite database** with 160K+ rows of realistic e-commerce data
- **Deterministic grading**: compare result sets for correctness + measure execution time
- **Meaningful partial credit**: valid SQL → correct results → fast → very fast
- A natural **difficulty progression** from simple filter pushdown to complex window functions

---

## 📐 Environment Specification

### Action Space

```python
@dataclass
class SqlSurgeonAction(Action):
    optimized_query: str   # The agent's rewritten SQL query
```

The agent submits a single optimized SQL query per step. The query must:
- Be valid SQLite syntax
- Produce identical results to the original query
- Execute faster than the original (ideally)

### Observation Space

On `reset()`, the agent receives:

| Field | Type | Description |
|---|---|---|
| `task_id` | str | Task identifier (e.g., "filter_scan") |
| `task_description` | str | Natural language description of what's slow |
| `original_query` | str | The slow SQL query to optimize |
| `schema_ddl` | str | Full CREATE TABLE statements |
| `sample_data` | str | First 5 rows per table (JSON) |
| `query_plan_original` | str | EXPLAIN QUERY PLAN output |
| `execution_time_original_ms` | float | Baseline execution time |
| `expected_row_count` | int | Number of rows the query should return |
| `hint` | str | Optimization hint (e.g., "Use JOINs instead of subqueries") |

After each `step()`, the agent also receives:

| Field | Type | Description |
|---|---|---|
| `is_valid_sql` | bool | Did the query parse? |
| `executed_successfully` | bool | Did it run without error? |
| `is_correct` | bool | Do results match the original? |
| `speedup` | float | original_time / optimized_time |
| `execution_time_optimized_ms` | float | Agent's query execution time |
| `error` | str | Error message (if any) |
| `attempts_remaining` | int | Steps left in the episode |
| `best_speedup_so_far` | float | Best speedup across all attempts |

### State

```python
@dataclass
class SqlSurgeonState(State):
    task_id: str
    attempts_used: int
    max_attempts: int = 5
    best_speedup: float
    cumulative_reward: float
```

---

## 🎮 Tasks

### Task 1: `filter_scan` — 🟢 Easy

**Problem**: Query uses a subquery that fetches ALL orders, then filters the result.
**Solution**: Push WHERE filters into the main query and use proper JOIN syntax.
**Expected speedup**: 2-5x

### Task 2: `subquery_to_join` — 🟡 Medium

**Problem**: Correlated subqueries recalculate AVG and COUNT for every product row (O(n×m)).
**Solution**: Replace correlated subqueries with a single JOIN + GROUP BY.
**Expected speedup**: 5-15x

### Task 3: `multi_table_optimize` — 🔴 Hard

**Problem**: Customer spending report with DISTINCT + 3 correlated subqueries + IN subquery.
**Solution**: Use CTEs for pre-aggregation, eliminate DISTINCT, collapse subqueries.
**Expected speedup**: 10-50x

### Task 4: `window_vs_self_join` — 💀 Expert

**Problem**: Customer ranking within cities using self-join counting (O(n²) per city).
**Solution**: Use RANK() window function with PARTITION BY (O(n log n)).
**Expected speedup**: 20-100x+

---

## 📊 Reward Function

The reward provides **meaningful partial credit** across the trajectory:

```
reward = 0.00

# Tier 1: Basic validity
+0.05  if query is valid SQL syntax
+0.10  if query executes without error

# Tier 2: Correctness
+0.35  if results match the original query exactly

# Tier 3: Performance (graduated)
+0.10  if speedup >= 1.5x
+0.15  if speedup >= 3.0x
+0.15  if speedup >= 5.0x
+0.10  if speedup >= 10.0x

# Penalty
0.00   if results are wrong (incorrect rewrite)
```

**Score range**: [0.0, 1.0] per step. Each task's score is the **best reward achieved** across up to 5 attempts.

---

## 📂 Database Schema

An e-commerce database with 4 tables and 161,000 rows:

```sql
customers  (10,000 rows) — id, first_name, last_name, email, city, country, created_at
products   ( 1,000 rows) — id, name, category, price, stock
orders    (100,000 rows) — id, customer_id, product_id, quantity, total_amount, status, created_at
reviews    (50,000 rows) — id, order_id, customer_id, product_id, rating, review_text, created_at
```

Data is generated deterministically (seed=42) for reproducible results.

---

## 🚀 Quick Start

### Install

```bash
pip install openenv-core
pip install -e .
```

### Run the server locally

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### Use the environment

```python
from sql_surgeon import SqlSurgeonEnv, SqlSurgeonAction

# Connect to a running server
with SqlSurgeonEnv(base_url="http://localhost:8000").sync() as env:
    result = env.reset(task_id="filter_scan")
    print(result.observation.metadata["original_query"])
    
    result = env.step(SqlSurgeonAction(optimized_query="SELECT ..."))
    print(f"Reward: {result.reward}, Speedup: {result.observation.metadata['speedup']}x")
```

### Run with Docker

```bash
# Build
docker build -t sql-surgeon:latest -f server/Dockerfile .

# Run
docker run -p 8000:8000 sql-surgeon:latest
```

### Run baseline inference

```bash
export HF_TOKEN=your_token
export LOCAL_IMAGE_NAME=sql-surgeon:latest
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
python inference.py
```

Required for submission:
- `inference.py` must stay at repository root.
- Use `OpenAI` client with `HF_TOKEN`, `API_BASE_URL`, and `MODEL_NAME`.
- Runtime logs must emit only `[START]`, `[STEP]`, and `[END]` lines in the required order.

---

## 📁 Project Structure

```
sql_surgeon/
├── __init__.py                     # Package exports
├── models.py                       # Action, Observation, State dataclasses
├── client.py                       # SqlSurgeonEnv client
├── inference.py                    # Baseline inference script
├── openenv.yaml                    # OpenEnv manifest
├── pyproject.toml                  # Dependencies
├── README.md                       # This file
├── .dockerignore                   # Docker exclusions
└── server/
    ├── __init__.py
    ├── app.py                      # FastAPI application
    ├── sql_surgeon_environment.py  # Core environment (step/reset/state)
    ├── database.py                 # SQLite setup + query execution
    ├── tasks.py                    # 7 optimization tasks (easy -> expert)
    ├── graders.py                  # Deterministic grading logic
    ├── requirements.txt            # Server dependencies
    └── Dockerfile                  # Container definition
└── scripts/
    └── validate-submission.sh      # Pre-submission validator
```

---

## 🏆 Baseline Scores

| Task | Difficulty | Baseline Score* | Expected Optimal |
|---|---|---|---|
| `filter_scan` | 🟢 Easy | ~0.65 | 1.0 |
| `subquery_to_join` | 🟡 Medium | ~0.50 | 1.0 |
| `multi_table_optimize` | 🔴 Hard | ~0.35 | 0.85+ |
| `window_vs_self_join` | 💀 Expert | ~0.15 | 0.70+ |

*Baseline scores using Qwen2.5-72B-Instruct with 5 attempts per task.

---

## 🔧 Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `API_BASE_URL` | No | `https://router.huggingface.co/v1` | LLM API endpoint |
| `MODEL_NAME` | No | `Qwen/Qwen2.5-72B-Instruct` | Model identifier |
| `HF_TOKEN` | Yes | — | HuggingFace API key |
| `LOCAL_IMAGE_NAME` | Yes* | — | Local docker image for `from_docker_image(...)` |
| `IMAGE_NAME` | No | — | Backward-compatible alias for local image name |

\*Required when running baseline locally from docker image.

## Pre-Submission Validation

Use the included validator script before submitting:

```bash
chmod +x scripts/validate-submission.sh
./scripts/validate-submission.sh https://your-space.hf.space .
```

It checks:
- HF Space responds on `POST /reset` with HTTP 200
- Docker image builds successfully
- `openenv validate` passes

---

## 📄 License

BSD-3-Clause — see [LICENSE](LICENSE).
