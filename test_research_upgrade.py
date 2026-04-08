from server.sql_surgeon_environment import SqlSurgeonEnvironment
from models import SqlSurgeonAction, SqlSurgeonActionType

def test_upgrade():
    env = SqlSurgeonEnvironment()
    
    # 1. Test Task: Semantics Hazard (The DISTINCT trap)
    print("\n--- Testing Task: semantics_hazard ---")
    obs = env.reset(task_id="semantics_hazard")
    print(f"Deceptive Signals: {obs.metadata['deceptive_hints']}")
    
    # Naive submission (no DISTINCT - Hallucination)
    action_naive = SqlSurgeonAction(
        action_type=SqlSurgeonActionType.SUBMIT,
        query="SELECT c.first_name, c.last_name, p.name as product_name FROM customers c JOIN orders o ON c.id = o.customer_id JOIN reviews r ON c.id = r.customer_id AND o.product_id = r.product_id JOIN products p ON o.product_id = p.id WHERE r.rating >= 4 ORDER BY 1, 2",
        confidence=1.0
    )
    result_n = env.step(action_naive)
    print(f"Naive (No DISTINCT) -> Reward: {result_n.reward}, Done: {result_n.done}")
    if "hallucination_info" in result_n.metadata:
        print(f"Hallucination Meta: {result_n.metadata['hallucination_info']}")

    # 2. Test Task: Index Trap (Confidence & Resistance)
    print("\n--- Testing Task: index_trap ---")
    env.reset(task_id="index_trap")
    
    # Tool call: Schema check
    action_schema = SqlSurgeonAction(action_type=SqlSurgeonActionType.CHECK_SCHEMA)
    result_s = env.step(action_schema)
    print(f"Tool (Schema) -> Reward: {result_s.reward}, Result length: {len(result_s.metadata['tool_result'])}")
    
    # Correct submission with high confidence and resistance
    action_correct = SqlSurgeonAction(
        action_type=SqlSurgeonActionType.SUBMIT,
        query="SELECT c.country, o.status, SUM(o.total_amount) as total FROM orders o JOIN customers c ON o.customer_id = c.id WHERE c.country = 'India' AND o.status = 'delivered' GROUP BY 1, 2",
        confidence=1.0
    )
    result_c = env.step(action_correct)
    print(f"Correct (Resistant) -> Reward: {result_c.reward}, Speedup: {result_c.metadata.get('speedup')}, Resistance Bonus: {result_c.metadata.get('deception_resistance')}")

if __name__ == "__main__":
    test_upgrade()
