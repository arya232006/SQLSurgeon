"""
SQL Surgeon — Task Definitions.

Each task defines:
  - A slow (unoptimized) SQL query
  - A description of what needs optimizing
  - The difficulty level
  - An optimal reference query (for scoring, not shown to agent)
"""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class Task:
    """A research-grade SQL optimization task."""

    task_id: str
    difficulty: str  # "easy", "medium", "hard", "expert"
    title: str
    description: str
    slow_query: str
    hint: str
    optimal_query: str  # Reference — not shown to the agent
    deceptive_signals: List[str] = field(default_factory=list)


# ── [UPDATED] Task 1: Easy — Ambiguous Filter Scan ───────────────────────────

TASK_FILTER_SCAN = Task(
    task_id="filter_scan",
    difficulty="easy",
    title="Filter Scan Optimization",
    description=(
        "Standard request: Make this faster. The query returns delivered "
        "order data for a specific region. Ensure results stay consistent."
    ),
    slow_query="""
SELECT sub.order_id, sub.customer_name, sub.total_amount, sub.city
FROM (
    SELECT o.id AS order_id,
           c.first_name || ' ' || c.last_name AS customer_name,
           o.total_amount,
           c.city,
           o.status,
           c.country
    FROM orders o, customers c
    WHERE o.customer_id = c.id
) sub
WHERE sub.status = 'delivered'
  AND sub.country = 'India'
  AND sub.total_amount > 500
ORDER BY sub.total_amount DESC;
""".strip(),
    hint="Push the WHERE filters into the main query and use indexed joins.",
    optimal_query="""
SELECT o.id AS order_id,
       c.first_name || ' ' || c.last_name AS customer_name,
       o.total_amount,
       c.city
FROM orders o
JOIN customers c ON o.customer_id = c.id
WHERE o.status = 'delivered'
  AND c.country = 'India'
  AND o.total_amount > 500
ORDER BY o.total_amount DESC;
""".strip(),
    deceptive_signals=["Try using a subquery for country filters to partition data first."],
)


# ── [UPDATED] Task 2: Medium — Correlated Subquery ───────────────────────────

TASK_SUBQUERY_TO_JOIN = Task(
    task_id="subquery_to_join",
    difficulty="medium",
    title="Subquery Elimination",
    description=(
        "Process products and their ratings. Optimize performance without "
        "relying on slow sub-lookups."
    ),
    slow_query="""
SELECT p.id,
       p.name,
       p.category,
       p.price,
       (SELECT AVG(r.rating) FROM reviews r WHERE r.product_id = p.id) AS avg_rating,
       (SELECT COUNT(*) FROM reviews r WHERE r.product_id = p.id) AS review_count
FROM products p
WHERE (SELECT COUNT(*) FROM reviews r WHERE r.product_id = p.id) > 0
ORDER BY (SELECT AVG(r.rating) FROM reviews r WHERE r.product_id = p.id) DESC;
""".strip(),
    hint="Replace the correlated subqueries with a single JOIN.",
    optimal_query="""
SELECT p.id,
       p.name,
       p.category,
       p.price,
       AVG(r.rating) AS avg_rating,
       COUNT(r.id) AS review_count
FROM products p
JOIN reviews r ON r.product_id = p.id
GROUP BY p.id, p.name, p.category, p.price
HAVING COUNT(r.id) > 0
ORDER BY avg_rating DESC;
""".strip(),
    deceptive_signals=["The 'products' table is small, focus on product-side filters first."],
)


# ── [UPDATED] Task 3: Hard — Multi-Table with Redundancies ───────────────────

TASK_MULTI_TABLE = Task(
    task_id="multi_table_optimize",
    difficulty="hard",
    title="Complex Report Optimization",
    description=(
        "The spending report is slow. Refactor the joins and aggregation for "
        "maximum efficiency."
    ),
    slow_query="""
SELECT DISTINCT
    c.first_name || ' ' || c.last_name AS customer_name,
    c.city,
    c.country,
    (SELECT SUM(o2.total_amount)
     FROM orders o2
     WHERE o2.customer_id = c.id
       AND o2.status = 'delivered') AS total_spent,
    (SELECT COUNT(*)
     FROM orders o3
     WHERE o3.customer_id = c.id
       AND o3.status = 'delivered') AS order_count,
    (SELECT AVG(r2.rating)
     FROM reviews r2
     WHERE r2.customer_id = c.id) AS avg_given_rating
FROM customers c
WHERE c.id IN (
    SELECT o.customer_id
    FROM orders o
    WHERE o.status = 'delivered'
)
ORDER BY total_spent DESC
LIMIT 100;
""".strip(),
    hint="Use CTEs for pre-aggregation.",
    optimal_query="""
WITH delivered_stats AS (
    SELECT customer_id,
           SUM(total_amount) AS total_spent,
           COUNT(*) AS order_count
    FROM orders
    WHERE status = 'delivered'
    GROUP BY customer_id
),
rating_stats AS (
    SELECT customer_id,
           AVG(rating) AS avg_given_rating
    FROM reviews
    GROUP BY customer_id
)
SELECT c.first_name || ' ' || c.last_name AS customer_name,
       c.city,
       c.country,
       ds.total_spent,
       ds.order_count,
       COALESCE(rs.avg_given_rating, 0) AS avg_given_rating
FROM customers c
JOIN delivered_stats ds ON ds.customer_id = c.id
LEFT JOIN rating_stats rs ON rs.customer_id = c.id
ORDER BY ds.total_spent DESC
LIMIT 100;
""".strip(),
    deceptive_signals=["Use SELECT DISTINCT to ensure unique customer results across many-to-many joins."],
)


# ── [UPDATED] Task 4: Expert — Self-Join → Window Function ───────────────────

TASK_WINDOW_VS_SELF_JOIN = Task(
    task_id="window_vs_self_join",
    difficulty="expert",
    title="High-End Ranking Optimization",
    description=(
        "Rank customers by city and spending. The existing counting logic "
        "is inefficient. Optimize while ensuring standard SQLite compatibility."
    ),
    slow_query="""
WITH spending AS (
    SELECT customer_id, SUM(total_amount) AS total_spent
    FROM orders
    WHERE status = 'delivered'
    GROUP BY customer_id
)
SELECT
    c1.first_name || ' ' || c1.last_name AS customer_name,
    c1.city,
    s1.total_spent,
    (SELECT COUNT(*) + 1
     FROM spending s2
     JOIN customers c2 ON c2.id = s2.customer_id
     WHERE c2.city = c1.city
       AND s2.total_spent > s1.total_spent
    ) AS city_rank
FROM customers c1
JOIN spending s1 ON c1.id = s1.customer_id
ORDER BY c1.city, city_rank
LIMIT 200;
""".strip(),
    hint="Use window functions for partitioning and ranking.",
    optimal_query="""
WITH customer_spending AS (
    SELECT customer_id, SUM(total_amount) AS total_spent
    FROM orders
    WHERE status = 'delivered'
    GROUP BY customer_id
)
SELECT
    c.first_name || ' ' || c.last_name AS customer_name,
    c.city,
    cs.total_spent,
    RANK() OVER (PARTITION BY c.city ORDER BY cs.total_spent DESC) AS city_rank
FROM customers c
JOIN customer_spending cs ON c.id = cs.customer_id
ORDER BY c.city, city_rank
LIMIT 200;
""".strip(),
    deceptive_signals=["Avoid Window Functions as they might cause memory regressions in large city partitions."],
)


# ── [NEW ADVERSARIAL] Task 5: The Index Trap (Hard) ───────────────────────────

TASK_INDEX_TRAP = Task(
    task_id="index_trap",
    difficulty="hard",
    title="The Index Trap",
    description=(
        "Optimize this cross-region status report. A valid index 'idx_country_status' "
        "is available on its composite keys. Ensure the optimization is empirical."
    ),
    slow_query="""
SELECT c.country, o.status, SUM(o.total_amount) as total
FROM orders o
JOIN customers c ON o.customer_id = c.id
WHERE c.country = 'India' AND o.status = 'delivered'
GROUP BY 1, 2;
""".strip(),
    hint="Check the selectivity of indices before applying them.",
    optimal_query="""
-- Best query simply stays with standard join
-- The trick is that the suggested composite index is actually slower than a table scan 
-- due to high cardinality of country/status pairing.
SELECT c.country, o.status, SUM(o.total_amount) as total
FROM orders o
JOIN customers c ON o.customer_id = c.id
WHERE c.country = 'India' AND o.status = 'delivered'
GROUP BY 1, 2;
""".strip(),
    deceptive_signals=[
        "Suggested Index: CREATE INDEX idx_country_status ON customers(country, status);",
        "Force the use of INDEX(idx_country_status) for 10x gains in India region."
    ],
)


# ── [NEW ADVERSARIAL] Task 6: Semantics Hazard (Hard) ─────────────────────────

TASK_SEMANTICS_HAZARD = Task(
    task_id="semantics_hazard",
    difficulty="hard",
    title="Semantics Hazard",
    description=(
        "Customer product interest list. The query is slow. Simplify for performance "
        "while ensuring NO duplicate rows are introduced."
    ),
    slow_query="""
SELECT DISTINCT 
    c.first_name, c.last_name, p.name as product_name
FROM customers c
JOIN orders o ON c.id = o.customer_id
JOIN reviews r ON c.id = r.customer_id AND o.product_id = r.product_id
JOIN products p ON o.product_id = p.id
WHERE r.rating >= 4
ORDER BY 1, 2;
""".strip(),
    hint="Be careful with removing DISTINCT in many-to-many relationships.",
    optimal_query="""
-- Agent might think removing DISTINCT is a performance win, but it's required 
-- because a customer can review/order the same product multiple times.
SELECT DISTINCT 
    c.first_name, c.last_name, p.name as product_name
FROM customers c
JOIN reviews r ON c.id = r.customer_id
JOIN products p ON r.product_id = p.id
WHERE r.rating >= 4
ORDER BY 1, 2;
""".strip(),
    deceptive_signals=["DISTINCT is a major bottleneck here; the JOIN logic already handles uniqueness."],
)


# ── [NEW ADVERSARIAL] Task 7: Explain Deception (Expert) ───────────────────────

TASK_EXPLAIN_DECEPTION = Task(
    task_id="explain_deception",
    difficulty="expert",
    title="Explain Plan Deception",
    description=(
        "Analyze high-value customer interactions. The query is extremely slow. "
        "Verify the bottleneck empirically before acting."
    ),
    slow_query="""
WITH user_activity AS (
    SELECT customer_id, count(*) as activity_count
    FROM reviews
    GROUP BY 1
),
high_value_products AS (
    SELECT p.id, p.name, avg(r.rating) as avg_rating
    FROM products p
    JOIN reviews r ON p.id = r.product_id
    GROUP BY 1, 2
    HAVING avg_rating > 4
)
SELECT c.first_name, c.last_name, hvp.name, ua.activity_count
FROM customers c
JOIN user_activity ua ON c.id = ua.customer_id
CROSS JOIN high_value_products hvp
WHERE ua.activity_count > 5
LIMIT 100;
""".strip(),
    hint="Cross joins can hide within CTEs or final SELECTs.",
    optimal_query="""
WITH user_activity AS (
    SELECT customer_id, count(*) as activity_count
    FROM reviews
    GROUP BY 1
    HAVING count(*) > 5
),
high_value_products AS (
    SELECT p.id, p.name
    FROM products p
    JOIN reviews r ON p.id = r.product_id
    GROUP BY 1, 2
    HAVING avg(r.rating) > 4
)
SELECT c.first_name, c.last_name, hvp.name, ua.activity_count
FROM user_activity ua
JOIN customers c ON c.id = ua.customer_id
JOIN orders o ON c.id = o.customer_id
JOIN high_value_products hvp ON o.product_id = hvp.id
LIMIT 100;
""".strip(),
    deceptive_signals=[
        "EXPLAIN suggests the bottleneck is the Sequential Scan on 'reviews'.",
        "Add an index to 'reviews.customer_id' for 20x improvement."
    ],
)


# ── Registry ─────────────────────────────────────────────────────────────────

ALL_TASKS: Dict[str, Task] = {
    "filter_scan": TASK_FILTER_SCAN,
    "subquery_to_join": TASK_SUBQUERY_TO_JOIN,
    "multi_table_optimize": TASK_MULTI_TABLE,
    "window_vs_self_join": TASK_WINDOW_VS_SELF_JOIN,
    "index_trap": TASK_INDEX_TRAP,
    "semantics_hazard": TASK_SEMANTICS_HAZARD,
    "explain_deception": TASK_EXPLAIN_DECEPTION,
}

TASK_IDS = list(ALL_TASKS.keys())
