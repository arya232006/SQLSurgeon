"""Quick sanity test for SQL Surgeon."""

from server.database import DatabaseManager
from server.tasks import ALL_TASKS
from server.graders import grade_query

db = DatabaseManager(seed=42)
db.initialize()

stats = db.get_table_stats()
print("=== Database Seeded ===")
for t, c in stats.items():
    print(f"  {t}: {c}")

print()
print("=== Grading Optimal Queries ===")
results = []
for tid, task in ALL_TASKS.items():
    db.reset()
    g = grade_query(db, task.slow_query, task.optimal_query)
    line = f"  {tid} [{task.difficulty}]: reward={g.reward} speedup={g.speedup}x correct={g.is_correct}"
    if g.error:
        line += f" ERROR={g.error[:60]}"
    print(line)
    results.append((tid, g))

db.close()

print()
all_ok = all(g.is_correct for _, g in results)
all_rewarded = all(g.reward > 0.3 for _, g in results)
print(f"All correct: {all_ok}")
print(f"All rewarded >0.3: {all_rewarded}")
if all_ok:
    print("PASS")
else:
    print("NEEDS FIXING")
    for tid, g in results:
        if not g.is_correct:
            print(f"  FAILED: {tid} - {g.error}")
