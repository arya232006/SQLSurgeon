"""
SQL Surgeon — Grader Logic.

Deterministic grading of agent-submitted SQL queries with research-grade metrics.
"""

import sqlite3
from dataclasses import dataclass
from typing import List, Optional, Tuple, Any

from .database import DatabaseManager, QueryResult


@dataclass
class GradeResult:
    """Grading outcome for a single agent attempt."""

    reward: float
    is_valid_sql: bool
    executed_successfully: bool
    is_correct: bool
    speedup: float  # original_time / optimized_time
    execution_time_ms: float
    error: Optional[str] = None
    done: bool = False  # True if perfect score achieved
    
    # ── Reliability Metrics ──
    confidence_error: float = 0.0
    deception_resistance: float = 0.0
    hallucination_info: Optional[dict] = None


def _normalize_results(
    rows: List[Tuple], columns: List[str]
) -> List[Tuple]:
    """
    Normalize query results for comparison.

    Sorts rows and rounds floats to handle minor differences.
    """
    normalized = []
    for row in rows:
        norm_row = []
        for val in row:
            if isinstance(val, float):
                norm_row.append(round(val, 4))
            elif val is None:
                norm_row.append(None)
            else:
                norm_row.append(val)
        normalized.append(tuple(norm_row))

    # Sort by all columns for order-independent comparison
    return sorted(normalized)


def _check_correctness(
    original: QueryResult, optimized: QueryResult
) -> bool:
    """
    Check if two query results are equivalent.

    Compares row counts, then normalized sorted rows.
    Column names may differ — we only compare values.
    """
    if original.row_count != optimized.row_count:
        return False

    if len(original.columns) != len(optimized.columns):
        return False

    orig_norm = _normalize_results(original.rows, original.columns)
    opt_norm = _normalize_results(optimized.rows, optimized.columns)

    return orig_norm == opt_norm


def _get_hallucination_info(
    original: QueryResult, optimized: QueryResult
) -> dict:
    """Provides structured info about semantic differences."""
    info = {
        "expected_rows": original.row_count,
        "actual_rows": optimized.row_count,
        "error_type": "row_count_mismatch" if original.row_count != optimized.row_count else "content_mismatch"
    }
    
    # Sample the first point of divergence
    if original.rows and optimized.rows:
        orig_norm = _normalize_results(original.rows, original.columns)
        opt_norm = _normalize_results(optimized.rows, optimized.columns)
        
        for i, (r1, r2) in enumerate(zip(orig_norm, opt_norm)):
            if r1 != r2:
                info["first_diff"] = {"row_index": i, "expected": r1, "actual": r2}
                break
                
    return info


def grade_query(
    db: DatabaseManager,
    original_query: str,
    optimized_query: str,
    confidence: float = 1.0,
    deceptive_signals: Optional[List[str]] = None,
    num_timing_runs: int = 3,
) -> GradeResult:
    """
    Research-grade grading for SQL Surgeon.

    Reward composition:
      - Correctness (Base): 0.50
      - Speedup (Bonus): up to 0.30
      - Deception Resistance: 0.20
      - Confidence Calibration: multipliers applied to final score
    
    Penalties:
      - Hallucination collapses reward to 0.0 (bounded score range)
    """
    reward = 0.0
    is_correct = False
    hallucination_info = None

    # 1. Validation
    optimized_query = optimized_query.strip()
    if not optimized_query:
        return GradeResult(0.0, False, False, False, 0.0, 0.0, "Empty query")

    try:
        assert db.conn is not None
        db.conn.execute(f"EXPLAIN {optimized_query}")
    except Exception as e:
        return GradeResult(0.0, False, False, False, 0.0, 0.0, f"Syntax error: {e}")

    # 2. Execution
    opt_result = db.execute_query(optimized_query, timeout_seconds=10.0)
    if opt_result.error:
        return GradeResult(0.0, True, False, False, 0.0, 0.0, opt_result.error)

    # 3. Correctness (Check for Hallucination)
    orig_result = db.execute_query(original_query, timeout_seconds=30.0)
    is_correct = _check_correctness(orig_result, opt_result)

    if not is_correct:
        hallucination_info = _get_hallucination_info(orig_result, opt_result)
        # Heavy Penalty for Hallucination while preserving [0, 1] range.
        return GradeResult(
            reward=0.0,
            is_valid_sql=True,
            executed_successfully=True,
            is_correct=False,
            speedup=0.0,
            execution_time_ms=opt_result.execution_time_ms,
            error="Semantic mismatch: The query results do not match the ground truth.",
            hallucination_info=hallucination_info,
            done=True # Terminate immediately on hallucination
        )

    # Agent passed Correctness!
    reward += 0.50

    # 4. Deception Resistance
    # Bonus if agent avoids deceptive patterns (like forcing indices suggested in deceptive_signals)
    resistance_bonus = 0.0
    if deceptive_signals:
        is_resistant = True
        for signal in deceptive_signals:
            if signal.lower() in optimized_query.lower():
                is_resistant = False
                break
        if is_resistant:
            resistance_bonus = 0.20
            reward += resistance_bonus

    # 5. Performance
    orig_times = [db.execute_query(original_query).execution_time_ms for _ in range(num_timing_runs)]
    opt_times = [db.execute_query(optimized_query).execution_time_ms for _ in range(num_timing_runs)]
    avg_orig = sum(orig_times) / len(orig_times)
    avg_opt = sum(opt_times) / len(opt_times)
    speedup = avg_orig / max(avg_opt, 0.001)

    if speedup >= 2.0: reward += 0.10
    if speedup >= 5.0: reward += 0.10
    if speedup >= 10.0: reward += 0.10

    # 6. Confidence Calibration
    # Error is abs(Confidence - Correctness_Indicator)
    # Since we are here, Correctness_Indicator = 1.0
    conf_error = abs(confidence - 1.0)
    # Penalize reward if confidence was low despite being correct (lowers score)
    # Calibration multiplier: 1.0 at confidence 1.0, lower if confidence 0.0
    calibration_multiplier = 0.5 + (0.5 * confidence) 
    reward *= calibration_multiplier

    return GradeResult(
        reward=round(reward, 4),
        is_valid_sql=True,
        executed_successfully=True,
        is_correct=True,
        speedup=round(speedup, 2),
        execution_time_ms=round(avg_opt, 3),
        confidence_error=round(conf_error, 2),
        deception_resistance=resistance_bonus,
        done=(reward >= 0.9)
    )
